"""Frozen synthetic prior-shift stress; does not fit or tune any model."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import softmax


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balance(p):
    factors = np.ones(p.shape[1])
    for _ in range(300):
        q = p * factors
        q /= q.sum(axis=1, keepdims=True)
        factors *= (len(p) / p.shape[1]) / q.sum(axis=0)
    q = p * factors
    return q / q.sum(axis=1, keepdims=True)


def metrics(y, pred):
    recalls = [float((pred[y == k] == k).mean()) for k in range(18)]
    return {"balanced_accuracy": float(np.mean(recalls)), "accuracy": float((pred == y).mean()), "recalls": recalls}


def main():
    started = time.monotonic()
    config_path = Path('configs/name-that-llm/v36.json')
    config = json.loads(config_path.read_text())
    root = Path('artifacts/improvement-v36')
    root.mkdir(exist_ok=True)
    assert not (root / 'report.json').exists(), 'Do not overwrite a completed experiment'
    (root / 'protocol.json').write_text(json.dumps(config, indent=2))
    paths = [config_path, Path(__file__)]
    frame_path = Path('artifacts/improvement-v22/kaggle-output/validation_frame.parquet')
    frame = pd.read_parquet(frame_path)
    paths.append(frame_path)
    labels = np.sort(frame.model.unique())
    assert len(labels) == 18 and len(frame) == 540
    assert (frame.groupby('group').size() == 18).all() and frame.group.nunique() == 30
    y = np.searchsorted(labels, frame.model.to_numpy())
    provenance = {}
    for version in [22, 25]:
        directory = Path(f'artifacts/improvement-v{version}/kaggle-output')
        train_path = directory / 'training_frame.parquet'
        valid_path = directory / 'validation_frame.parquet'
        train, valid = pd.read_parquet(train_path), pd.read_parquet(valid_path)
        assert np.array_equal(valid.id, frame.id)
        assert not set(train.id.astype(str)) & set(frame.id.astype(str))
        assert not set(train.group.astype(str)) & set(frame.group.astype(str))
        paths.extend([train_path, valid_path])
        provenance[str(version)] = {'fit_rows': len(train), 'heldout_rows': len(valid), 'id_overlap': 0, 'group_overlap': 0}
    fold_path = Path('artifacts/baseline-v1/folds.csv')
    folds = pd.read_csv(fold_path, dtype={'id': str})
    assert set(folds.loc[folds.fold == 0, 'id']) == set(frame.id.astype(str))
    assert not set(folds.loc[folds.fold != 0, 'prompt_id']) & set(folds.loc[folds.fold == 0, 'prompt_id'])
    paths.append(fold_path)
    arrays = {}
    for name, path in {
        'sparse': 'artifacts/improvement-v17/uniform_pool_pilot.npz',
        'v22_supcon': 'artifacts/improvement-v22/kaggle-output/supcon_final_validation.npz',
        'v25_ce': 'artifacts/improvement-v25/kaggle-output/ce_final_validation.npz',
    }.items():
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z['ids'], frame.id.astype(str))
            assert np.array_equal(z['labels'], labels)
            arrays[name] = z['scores'].copy()
            assert arrays[name].shape == (540, 18) and np.isfinite(arrays[name]).all()
        paths.append(Path(path))
    rng = np.random.default_rng(config['seed'])
    orders = [rng.permutation(np.flatnonzero(y == k)) for k in range(18)]
    records, saved = [], {}
    for scenario in config['scenarios']:
        for rotation in range(1 if scenario['low_count'] == 30 else config['rotations']):
            high = {(rotation + k) % 18 for k in range(9)}
            idx = np.sort(np.concatenate([orders[k][:30 if k in high else scenario['low_count']] for k in range(18)]))
            for name in ['v22_supcon', 'v25_ce']:
                p = softmax((arrays['sparse'] + .15 * arrays[name])[idx], axis=1)
                q = balance(p)
                assert np.isfinite(q).all() and np.allclose(q.sum(0), len(idx) / 18)
                before, after = p.argmax(1), q.argmax(1)
                raw, adjusted = metrics(y[idx], before), metrics(y[idx], after)
                key = f"{scenario['name']}_{rotation}_{name}"
                saved[key + '_indices'] = idx
                saved[key + '_raw'] = before
                saved[key + '_balanced'] = after
                records.append({'key': key, 'scenario': scenario['name'], 'reference': name, 'rotation': rotation,
                    'rows': len(idx), 'raw': raw, 'equal_mass': adjusted,
                    'delta_balanced_accuracy': adjusted['balanced_accuracy'] - raw['balanced_accuracy'],
                    'true_counts': np.bincount(y[idx], minlength=18).tolist(),
                    'raw_predicted_counts': np.bincount(before, minlength=18).tolist(),
                    'balanced_predicted_counts': np.bincount(after, minlength=18).tolist()})
    summary = {}
    for scenario in config['scenarios']:
        for name in ['v22_supcon', 'v25_ce']:
            rows = [r for r in records if r['scenario'] == scenario['name'] and r['reference'] == name]
            deltas = [r['delta_balanced_accuracy'] for r in rows]
            summary[f"{scenario['name']}/{name}"] = {
                'raw_mean_ba': float(np.mean([r['raw']['balanced_accuracy'] for r in rows])),
                'equal_mass_mean_ba': float(np.mean([r['equal_mass']['balanced_accuracy'] for r in rows])),
                'delta_mean': float(np.mean(deltas)), 'delta_min': min(deltas), 'delta_max': max(deltas),
                'negative_rotations': sum(d < 0 for d in deltas), 'rotations': len(rows)}
    np.savez_compressed(root / 'predictions.npz', truth=y, ids=frame.id.to_numpy(dtype=str), labels=labels.astype(str), **saved)
    report = {'protocol': config, 'provenance': provenance, 'summary': summary, 'records': records,
        'input_sha256': {str(p): digest(p) for p in paths}, 'seconds': time.monotonic() - started,
        'decision': 'Diagnostic only: do not change full incumbent balancing based solely on proxy synthetic stress.'}
    (root / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'summary': summary, 'seconds': report['seconds']}, indent=2))


if __name__ == '__main__':
    main()
