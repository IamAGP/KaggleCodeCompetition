"""V41 delivery probe: unchanged DeBERTa pooling/head, CPU versus TPU.

Uses the installed Kaggle stack; no training, installs, or leaderboard submission.
Official device reference: pytorch/xla v2.8.0 docs/source/learn/pytorch-on-xla-devices.md.
"""
import os
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['HF_HUB_OFFLINE'] = '1'
import json
import signal
import time
import traceback
from pathlib import Path

started = time.monotonic()
report = {'experiment': 'v41-deberta-tpu-delivery-probe', 'passed': False,
          'scope': 'Installed-stack CPU/TPU parity; not CUDA parity or full ensemble eligibility'}

def save():
    report['elapsed_seconds'] = time.monotonic() - started
    Path('v41-probe.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)

def timeout(*_):
    raise TimeoutError('V41 600-second cap')

signal.signal(signal.SIGALRM, timeout)
signal.alarm(600)
try:
    import importlib.metadata
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModel, AutoTokenizer
    from safetensors.torch import load_file
    report['versions'] = {x: importlib.metadata.version(x) for x in ['torch', 'torch-xla', 'transformers']}
    torch.set_num_threads(4)
    roots = list(Path('/kaggle/input').rglob('deberta_v3_large/meta.json'))
    assert len(roots) == 1, roots
    root = roots[0].parent
    meta = json.loads(roots[0].read_text())
    train_paths = [p for p in Path('/kaggle/input').rglob('train.csv') if 'name-that-llm' in str(p)]
    assert len(train_paths) == 1, train_paths
    frame = pd.read_csv(train_paths[0]).iloc[:2]
    tok = AutoTokenizer.from_pretrained(root / 'encoder', local_files_only=True)
    model = AutoModel.from_pretrained(root / 'encoder', dtype=torch.float32, local_files_only=True).eval()
    head = load_file(root / 'head.safetensors')
    weight, bias = head['weight'].float().numpy(), head['bias'].float().numpy()
    batch = tok(frame.text.tolist(), truncation=True, max_length=meta['max_len'],
                padding='max_length', return_tensors='pt')
    report['shape'] = list(batch['input_ids'].shape)
    report['labels'] = meta['labels']
    def pool(inputs):
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        return (model(**inputs).last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
    save()
    t = time.monotonic()
    with torch.inference_mode():
        cpu = pool(batch).numpy()
    report['cpu_seconds'] = time.monotonic() - t
    save()
    import torch_xla
    import torch_xla.runtime as xr
    report['device_type'] = xr.device_type()
    assert report['device_type'] == 'TPU'
    device = torch_xla.device()
    model = model.to(device)
    xb = {k: v.to(device) for k, v in batch.items()}
    timings = []
    with torch.inference_mode():
        for iteration in range(4):
            t = time.monotonic()
            actual = pool(xb).cpu().numpy()  # CPU copy synchronizes lazy execution.
            timings.append(time.monotonic() - t)
            report['tpu_seconds_including_first_compile'] = timings
            save()
    cpu_logits, tpu_logits = cpu @ weight.T + bias, actual @ weight.T + bias
    cosine = (cpu * actual).sum(1) / (np.linalg.norm(cpu, axis=1) * np.linalg.norm(actual, axis=1))
    report.update(min_feature_cosine=float(cosine.min()),
                  max_logit_abs_error=float(np.max(np.abs(cpu_logits-tpu_logits))),
                  prediction_agreement=float(np.mean(cpu_logits.argmax(1)==tpu_logits.argmax(1))),
                  steady_seconds_per_row=float(np.median(timings[1:])/len(frame)),
                  single_component_4000_row_seconds=float(np.median(timings[1:])*2000))
    report['passed'] = bool(cosine.min() >= .999 and report['max_logit_abs_error'] <= .05
                            and report['prediction_agreement'] == 1)
    np.savez_compressed('v41-parity.npz', ids=frame.id.to_numpy(dtype=str),
                        cpu_features=cpu, tpu_features=actual, cpu_logits=cpu_logits, tpu_logits=tpu_logits)
    save()
except Exception:
    report['exception'] = traceback.format_exc()
    save()
    raise
finally:
    signal.alarm(0)
