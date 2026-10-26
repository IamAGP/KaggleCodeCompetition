"""Read-only diagnostic of the credited baseline's fixed class balancing.

Uses existing held-out scores; no fitting, weight search, or gate changes.
Balancing follows Nishant Dahal's Two-stage LoRA part two v2 (credited V16).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax


def balance(probabilities):
    factors = np.ones(probabilities.shape[1])
    target = len(probabilities) / probabilities.shape[1]
    for _ in range(300):
        adjusted = probabilities * factors
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        factors *= target / adjusted.sum(axis=0)
    adjusted = probabilities * factors
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def main():
    destination = Path('artifacts/diagnostics/balance-transfer-20261026.json')
    assert not destination.exists(), 'Completed diagnostic must not be overwritten'
    paths = {
        'sparse': 'artifacts/improvement-v17/uniform_pool_pilot.npz',
        'v19_small_4b': 'artifacts/improvement-v19/kaggle-output/supcon_final_validation.npz',
        'v22_large_4b': 'artifacts/improvement-v22/kaggle-output/supcon_final_validation.npz',
        'v25_8b_ce': 'artifacts/improvement-v25/kaggle-output/ce_final_validation.npz',
        'v25_8b_supcon': 'artifacts/improvement-v25/kaggle-output/supcon_final_validation.npz',
    }
    frame = pd.read_parquet('artifacts/improvement-v22/kaggle-output/validation_frame.parquet')
    labels = np.sort(frame.model.unique())
    scores = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as saved:
            assert np.array_equal(saved['labels'], labels)
            assert np.array_equal(saved['ids'], frame.id.astype(str))
            scores[name] = saved['scores'].copy()
            assert scores[name].shape == (540, 18) and np.isfinite(scores[name]).all()
    assert (frame.model.value_counts() == 30).all()
    truth = frame.model.to_numpy()
    metrics, correct = {}, {}
    for name, values in scores.items():
        combined = values if name == 'sparse' else scores['sparse'] + .15 * values
        probabilities = softmax(combined, axis=1)
        adjusted = balance(probabilities)
        assert np.isfinite(adjusted).all()
        assert np.allclose(adjusted.sum(1), 1)
        before = labels[probabilities.argmax(1)] == truth
        after = labels[adjusted.argmax(1)] == truth
        correct[name] = {'before': before, 'after': after}
        metrics[name] = {
            'unbalanced_accuracy': float(before.mean()), 'balanced_accuracy': float(after.mean()),
            'corrected': int((after & ~before).sum()), 'newly_wrong': int((before & ~after).sum()),
            'max_column_mass_error': float(np.max(np.abs(adjusted.sum(0) - 30))),
            'median_weighted_score_std_ratio': None if name == 'sparse' else
                float(np.median(.15 * values.std(1) / scores['sparse'].std(1))),
        }
    groups = frame.group.to_numpy(); unique = np.unique(groups)
    assert len(unique) == 30 and all((groups == g).sum() == 18 for g in unique)
    draws = np.random.default_rng(20261026).integers(0, 30, (10000, 30))
    comparisons = {}
    for setting in ['before', 'after']:
        delta = correct['v22_large_4b'][setting].astype(float) - correct['v19_small_4b'][setting]
        group_delta = np.array([delta[groups == g].mean() for g in unique])
        comparisons[setting] = {'v22_minus_v19': float(delta.mean()),
                               'prompt_bootstrap_95_interval': np.quantile(group_delta[draws].mean(1), [.025,.975]).tolist()}
    result = {
        'kind': 'post-hoc diagnostic, not a new selection experiment',
        'rows': 540, 'prompts': 30, 'iterations': 300, 'component_weight': .15,
        'credit': 'Nishant Dahal, Two-stage LoRA part two v2; same equal-class-mass balancing as credited V16',
        'metrics': metrics, 'comparisons': comparisons,
        'input_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths.values()},
        'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations': [
            'Sparse proxy still differs from full submitted ensemble; this cannot establish the cause of V23 regression',
            'Held-out classes are exactly balanced; hidden class balance is an assumption',
            'Repeatedly inspected development fold; bootstrap does not remove selection bias',
            'No new training, model selection, revised blending weights, or changes to V30 promotion gates',
            'Balanced here describes probability postprocessing; because labels have equal counts, ordinary accuracy equals the competition balanced-accuracy metric',
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps({'metrics': metrics, 'comparisons': comparisons}, indent=2))


if __name__ == '__main__':
    main()
