"""Independently check downloaded V21/V23 inference before submission."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probabilities(scores):
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p


def balanced(p):
    if len(p) < 50:
        return p
    scale = np.ones(p.shape[1])
    for _ in range(300):
        rows = p * scale
        rows /= rows.sum(axis=1, keepdims=True)
        scale *= (len(p) / p.shape[1]) / rows.sum(axis=0)
    rows = p * scale
    return rows / rows.sum(axis=1, keepdims=True)


def audit(version):
    root = Path(f'artifacts/improvement-{version}')
    output = root / 'inference-output'
    plan = json.loads((root / 'inference-plan.json').read_text())
    report = json.loads((output / f'{version}_audit.json').read_text())
    assert report['experiment'] == version
    assert report['asset_sha256'] == plan['inference_asset_sha256']
    for relative, digest in report['asset_sha256'].items():
        assert sha(root / 'kaggle-output' / relative) == digest, relative
    assert sha(output / 'component_v21.py') == plan['source_sha256']['component_v21.py']
    notebook = Path(f'notebooks/improvement-{version}/inference.ipynb')
    assert sha(notebook) == plan['inference_notebook_sha256']
    test = pd.read_csv('data/name-that-llm/test.csv', dtype={'id': str})
    submission = pd.read_parquet(output / 'submission.parquet')
    assert submission.columns.tolist() == ['id', 'model']
    assert submission.id.tolist() == test.id.tolist() and submission.id.is_unique
    with np.load(output / f'{version}_scores.npz', allow_pickle=False) as scores:
        ids, labels = scores['ids'], scores['labels']
        assert ids.tolist() == test.id.tolist()
        assert len(labels) == len(set(labels)) == 18
        assert set(labels) == set(plan['labels'])
        arrays = [scores[key] for key in ('incumbent_scores', 'component_scores', 'combined_scores')]
    base, component, combined = arrays
    assert all(x.shape == (len(test), 18) and np.isfinite(x).all() for x in arrays)
    weight = plan['blend']['new_member_log_probability_weight']
    assert weight == report['component_weight'] == 0.15
    reconstructed = base + weight * np.log(np.clip(probabilities(component), 1e-9, 1))
    np.testing.assert_allclose(combined, reconstructed, rtol=1e-10, atol=1e-10)
    predicted = balanced(probabilities(reconstructed))
    assert np.isfinite(predicted).all()
    assert submission.model.tolist() == labels[predicted.argmax(axis=1)].tolist()
    expected = {'svc': 1.0, 'lora_qwen3_14b': 0.4, 'lora_qwen25_14b': 0.2,
                'deberta': 0.25, f'contrastive_{version}': 0.15}
    members = report['members']
    weights = {m['member']: m['weight'] for m in members}
    assert len(weights) == len(members)
    assert all(weights.get(name) == value for name, value in expected.items())
    optional = {'lora_qwen3_14b twin': 0.1, 'lora_qwen25_14b twin': 0.1}
    assert all(name in optional and weight == optional[name]
               for name, weight in weights.items() if name not in expected)
    assert all(m['rows'] == len(test) for m in members if m['member'] in expected)
    assert all(0 < m['rows'] <= len(test) for m in members)
    changed = int((balanced(probabilities(base)).argmax(axis=1) != predicted.argmax(axis=1)).sum())
    assert changed == report['changed_predictions_vs_v16']
    assert report['rows'] == len(test) and 0 < report['elapsed_seconds'] < 9 * 3600
    result = dict(verified=True, rows=len(test), runtime_seconds=report['elapsed_seconds'],
                  member_weights=weights, changed_visible_predictions=changed,
                  submission_sha256=sha(output / 'submission.parquet'),
                  checks=['ordered IDs/schema', '18 labels', 'member weights', 'finite scores',
                          'independent blend and final predictions', 'source/checkpoint hashes'],
                  limitations=['Visible rows establish format/runtime, not hidden accuracy or runtime'])
    (root / 'independent-inference-audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=['v21', 'v23'], required=True)
    audit(parser.parse_args().version)
