"""New, matched-fold comparison of fixed v10 and nonlinear Qwen probes."""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from name_that_llm.qwen_probe_v11 import FrozenV10Probe, row_z
from name_that_llm.qwen_probe_v12 import QwenDiscriminantProbe
from name_that_llm.classifier_v7 import combine_evidence, RECIPES
from run_embeddings_v5 import load_incumbent, read_json, bootstrap_delta
from run_embeddings_v7 import verify_selection


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


def run(mode):
    root = Path('artifacts/improvement-v13')
    plan = read_json('configs/name-that-llm/v13.json')
    assert sha('configs/name-that-llm/v13.json') == sha(root / 'frozen-plan.json')
    for p, expected in plan['inputs'].items():
        assert sha(p) == expected, p
    selected = read_json(root / 'development/result.json') if mode == 'robustness' else None
    if selected is not None:
        assert selected['passes'], 'Development gate failed'
    out = root / mode
    out.mkdir(exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(out / 'debug.log'), logging.StreamHandler()])
    before = {'runner': sha(__file__), 'probe': sha('src/name_that_llm/qwen_probe_v12.py')}
    verify_selection(Path('artifacts/improvement-v7/development/selected.json'))
    v7root = Path('artifacts/improvement-v7') / mode
    for rec in read_json(v7root / 'artifacts.json')['files']:
        assert sha(v7root / rec['path']) == rec['sha256']
    train, folds, groups, v4, _, _, paths = load_incumbent(read_json('configs/name-that-llm/v7.json'), mode)
    report = read_json('artifacts/improvement-v10/kaggle-output/v9_report.json')
    assert report['train_ids'] == train.id.tolist()
    labels = sorted(train.model.unique())
    assert report['labels'] == labels
    assert report['members'] == ['L24_C0.02', 'L9_C0.02', 'L24_C0.2']
    X = np.load('artifacts/improvement-v10/kaggle-output/v9_train_features.npy', allow_pickle=False)
    assert X.shape == (len(train), 5, 2560) and np.isfinite(X).all()
    y = train.model.map({x: i for i, x in enumerate(labels)}).to_numpy()
    v7 = combine_evidence(np.load(v7root / 'embedding_oof_evidence.npy'), v4, RECIPES['styledistance_v4_half'])
    choices = plan['candidates'] if selected is None else {selected['selected']: plan['candidates'][selected['selected']]}
    qwen = np.zeros_like(v7)
    kernels = {name: np.zeros_like(v7) for name in choices}
    audit = []
    save(out / 'definition.json', {'plan_sha256': sha(root / 'frozen-plan.json'), 'sources': before,
        'mode': mode, 'labels': labels, 'row_ids': train.id.tolist(), 'inputs': plan['inputs'],
        'v7_evidence_sha256': sha(v7root / 'embedding_oof_evidence.npy'),
        'folds_sha256': sha(read_json('configs/name-that-llm/v7.json')[mode]['folds'])})
    with threadpool_limits(limits=2), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for fold in sorted(np.unique(folds)):
            t = time.monotonic()
            a, b = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            assert not set(groups[a]) & set(groups[b])
            base = FrozenV10Probe().fit(X[a], y[a])
            qwen[b] = base.decision_function(X[b])
            logging.info('fold %s fixed v10 Qwen reference %.1fs', fold, time.monotonic()-t)
            for name, C in choices.items():
                probe = QwenDiscriminantProbe(C).fit(X[a], y[a])
                kernels[name][b] = probe.decision_function(X[b])
                logging.info('fold %s %s cumulative %.1fs', fold, name, time.monotonic()-t)
            audit.append({'fold': int(fold), 'train_indexes_sha256': hashlib.sha256(a.tobytes()).hexdigest(),
                          'heldout_indexes_sha256': hashlib.sha256(b.tobytes()).hexdigest(),
                          'groups_disjoint': True})
    incumbent = .5*row_z(v7) + .5*row_z(qwen)
    np.save(out / 'v10_evidence.npy', incumbent)
    np.save(out / 'qwen_reference.npy', qwen)
    ba = lambda s: float(balanced_accuracy_score(y, s.argmax(1)))
    fold_ba = lambda s: [float(balanced_accuracy_score(y[folds == f], s[folds == f].argmax(1))) for f in sorted(np.unique(folds))]
    baseline = {'balanced_accuracy': ba(incumbent), 'folds': fold_ba(incumbent)}
    results = {}
    for name, scores in kernels.items():
        candidate = .5*row_z(incumbent) + .5*row_z(scores)
        np.save(out / (name + '_kernel.npy'), scores)
        np.save(out / (name + '_blend.npy'), candidate)
        metric = {'balanced_accuracy': ba(candidate), 'folds': fold_ba(candidate), 'kernel_alone': ba(scores)}
        metric['gain'] = metric['balanced_accuracy'] - baseline['balanced_accuracy']
        metric['fold_wins'] = sum(a > b + 1e-12 for a,b in zip(metric['folds'], baseline['folds']))
        results[name] = metric
    chosen = max(results, key=lambda k: (results[k]['balanced_accuracy'], choices[k] == .9))
    gain = results[chosen]['gain']
    passed = gain >= plan['gates'][mode + '_gain']
    if mode == 'robustness':
        passed = passed and results[chosen]['fold_wins'] >= plan['gates']['robustness_fold_wins']
    result = {'mode': mode, 'selected': chosen, 'passes': passed, 'baseline': baseline, 'results': results,
              'source_sha256': before, 'plan_sha256': sha(root / 'frozen-plan.json'), 'limitations': plan['limitations']}
    assert before == {'runner': sha(__file__), 'probe': sha('src/name_that_llm/qwen_probe_v12.py')}
    for p, expected in plan['inputs'].items(): assert sha(p) == expected
    save(out / 'fit-audit.json', audit)
    save(out / 'result.json', result)
    save(out / 'artifact-manifest.json', {'files': [{'path':p.name,'sha256':sha(p),'size_bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['development', 'robustness'])
    run(parser.parse_args().mode)
