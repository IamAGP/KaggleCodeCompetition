"""Build v9: frozen Qwen3-4B-Base hidden-state probe with in-notebook prompt-grouped CV.

The notebook extracts mean-pooled hidden states from fixed layers, evaluates a fixed
candidate grid of logistic probes with five-fold GroupKFold by prompt_id, refits the
best candidate on all training rows, and saves OOF/test probabilities for later blending.
"""
import argparse
import json
from pathlib import Path

import nbformat

LAYERS = [9, 18, 24, 30, 36]
C_GRID = [0.02, 0.2]

CODE = r'''
import gc, glob, json, os, time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

T0 = time.time()
LAYERS, C_GRID = __LAYERS__, __C_GRID__
OUT = '/kaggle/working'
def log(msg):
    print(f'[{time.strftime("%H:%M:%S")} +{time.time()-T0:7.1f}s] {msg}', flush=True)

DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
train, test = pd.read_csv(DATA + '/train.csv'), pd.read_csv(DATA + '/test.csv')
test['text'] = test['text'].fillna('').astype(str)
LABELS = sorted(train['model'].unique())
y = train['model'].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
cfg = [p for p in glob.glob('/kaggle/input/**/config.json', recursive=True) if '4b-base' in p.lower()]
assert len(cfg) >= 1, 'attach qwen-lm/qwen-3/Transformers/4b-base'
MODEL_DIR = os.path.dirname(sorted(cfg)[0])
log(f'train={train.shape} test={test.shape} labels={len(LABELS)} model={MODEL_DIR} gpus={torch.cuda.device_count()} '
    f'{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}')

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float16, device_map='auto').eval()
n_layers = model.config.num_hidden_layers
assert max(LAYERS) <= n_layers, (LAYERS, n_layers)
log(f'model loaded: layers={n_layers} hidden={model.config.hidden_size}')

@torch.no_grad()
def features(texts, name, batch=16):
    texts = list(texts)
    order = np.argsort([len(t) for t in texts])
    feats = np.zeros((len(texts), len(LAYERS), model.config.hidden_size), dtype=np.float32)
    for b, start in enumerate(range(0, len(texts), batch)):
        idx = order[start:start + batch]
        enc = tok([texts[i] for i in idx], return_tensors='pt', padding=True, truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        hs = model(**enc, output_hidden_states=True).hidden_states
        mask = enc['attention_mask'].unsqueeze(-1)
        for j, layer in enumerate(LAYERS):
            h = hs[layer].float()
            m = mask.to(h.device).float()
            feats[idx, j] = ((h * m).sum(1) / m.sum(1)).cpu().numpy()
        if b % 25 == 0:
            log(f'{name}: batch {b} rows {min(start + batch, len(texts))}/{len(texts)}')
    np.save(f'{OUT}/v9_{name}_features.npy', feats)
    return feats

Xtr = features(train['text'], 'train')
Xte = features(test['text'], 'test')
del model; gc.collect(); torch.cuda.empty_cache()
log('features done; starting grouped CV')

def probe(C):
    return LogisticRegression(C=C, max_iter=500, n_jobs=-1)

folds = list(GroupKFold(n_splits=5).split(Xtr, y, groups=train['prompt_id']))
results, oof_store = {}, {}
for j, layer in enumerate(LAYERS):
    for C in C_GRID:
        oof = np.zeros((len(y), len(LABELS)))
        for k, (a, b) in enumerate(folds):
            sc = StandardScaler().fit(Xtr[a, j])
            oof[b] = probe(C).fit(sc.transform(Xtr[a, j]), y[a]).predict_proba(sc.transform(Xtr[b, j]))
        key = f'L{layer}_C{C}'
        results[key] = float(balanced_accuracy_score(y, oof.argmax(1)))
        oof_store[key] = oof
        log(f'CV {key}: balanced_accuracy={results[key]:.4f}')
top3 = sorted(results, key=results.get, reverse=True)[:3]
ens = np.mean([np.log(np.clip(oof_store[k], 1e-12, 1)) for k in top3], axis=0)
results['top3_logmean'] = float(balanced_accuracy_score(y, ens.argmax(1)))
log(f'CV top3_logmean {top3}: {results["top3_logmean"]:.4f}')
best = max(results, key=results.get)
log(f'SELECTED {best} cv={results[best]:.4f}')

members = top3 if best == 'top3_logmean' else [best]
test_log = []
for key in members:
    layer, C = int(key.split('_')[0][1:]), float(key.split('_C')[1])
    j = LAYERS.index(layer)
    sc = StandardScaler().fit(Xtr[:, j])
    test_log.append(np.log(np.clip(probe(C).fit(sc.transform(Xtr[:, j]), y).predict_proba(sc.transform(Xte[:, j])), 1e-12, 1)))
test_scores = np.mean(test_log, axis=0)
oof_scores = ens if best == 'top3_logmean' else np.log(np.clip(oof_store[best], 1e-12, 1))
np.save(f'{OUT}/v9_oof_logprob.npy', oof_scores); np.save(f'{OUT}/v9_test_logprob.npy', test_scores)
sub = pd.DataFrame({'id': test['id'].astype(str), 'model': np.asarray(LABELS)[test_scores.argmax(1)]})
assert sub['id'].is_unique and len(sub) == len(test) and sub['model'].isin(LABELS).all()
sub.to_parquet(f'{OUT}/submission.parquet', index=False)
json.dump({'cv': results, 'selected': best, 'members': members, 'labels': LABELS, 'layers': LAYERS, 'c_grid': C_GRID,
           'train_ids': train['id'].astype(str).tolist(), 'seconds': time.time() - T0},
          open(f'{OUT}/v9_report.json', 'w'), indent=2)
log(f'done: {sub.shape} -> submission.parquet')
'''


def build(folder):
    code = CODE.replace("__LAYERS__", repr(LAYERS)).replace("__C_GRID__", repr(C_GRID))
    compile(code, "<v9>", "exec")
    nb = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(
            "# Name That LLM — Frozen Qwen3-4B Probe V9\n\nMean-pooled hidden states from fixed layers of "
            "Qwen3-4B-Base (no fine-tuning) feed standardized logistic probes. A fixed grid "
            f"(layers {LAYERS}, C {C_GRID}, plus a top-3 log-mean) is scored by five-fold GroupKFold on "
            "prompt_id; the best is refit on all rows. Selection on the same CV makes the reported score "
            "optimistic. GPU; internet disabled; competition data only."),
        nbformat.v4.new_code_cell(code.strip())],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbformat.validate(nb)
    folder.mkdir(parents=True, exist_ok=True)
    name = "name-that-llm-qwen3-probe-v9.ipynb"
    nbformat.write(nb, folder / name)
    meta = {"id": "adithyagiri/name-that-llm-qwen3-probe-v9", "title": "Name That LLM Qwen3 Probe V9",
            "code_file": name, "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "competition_sources": ["name-that-llm"], "dataset_sources": [], "kernel_sources": [],
            "model_sources": ["qwen-lm/qwen-3/Transformers/4b-base/1"]}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return folder / name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("notebooks/improvement-v9"))
    print(build(parser.parse_args().output_dir))
