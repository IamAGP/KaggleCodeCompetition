"""Frozen paired raw and fixed-blend V46 gates; fail closed on misalignment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_v43 import balanced, softmax


def evaluate(plan, output, root):
    for relative, digest in plan['evaluation_sha256'].items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == digest, relative
    comparisons, hashes, fold_prompts = [], {}, {}
    for fold in plan['folds']:
        base = np.load(root/f'artifacts/improvement-v34/kaggle-output/8b_ce/fold-{fold}/ce_final_validation.npz', allow_pickle=False)
        bp = softmax(base['scores'])
        for arm in plan['arms']:
            folder = output/arm/f'fold-{fold}'
            assert json.loads((folder/'run-config.json').read_text()) == dict(plan=plan, arm=arm, fold=fold)
            assert json.loads((folder/'result.json').read_text())['completed'] is True
            path = folder/'validation.npz'
            c = np.load(path, allow_pickle=False)
            ref = np.load(root/f'artifacts/improvement-v42/final-validation/{arm}/fold-{fold}/validation.npz', allow_pickle=False)
            assert all(np.array_equal(c[k], ref[k]) for k in ['ids','labels','targets','prompts'])
            assert all(np.array_equal(c[k], base[k]) for k in ['ids','labels'])
            assert len(set(c['ids'])) == len(c['ids'])
            assert list(c['labels']) == plan['labels']
            assert c['logits'].shape == bp.shape
            assert np.isfinite(c['logits']).all() and np.isfinite(bp).all()
            assert np.array_equal(np.unique(c['targets']), np.arange(len(plan['labels'])))
            split = json.loads((folder/'split.json').read_text())
            assert split['validation_ids'] == list(c['ids'])
            assert set(split['fit_ids']).isdisjoint(c['ids'])
            fold_prompts[fold] = set(c['prompts'])
            hashes[f'{arm}/fold-{fold}'] = hashlib.sha256(path.read_bytes()).hexdigest()
            score = lambda p: balanced(c['targets'], p.argmax(1), len(plan['labels']))
            candidate, reference, baseline = score(c['logits']), score(ref['logits']), score(bp)
            blend = score(0.85*bp + 0.15*softmax(c['logits'].astype(np.float64)))
            comparisons.append(dict(arm=arm, fold=fold, candidate=candidate, reference=reference,
                                    base=baseline, blend=blend, raw_gain=candidate-reference, blend_gain=blend-baseline))
    for a in fold_prompts:
        for b in fold_prompts:
            if a != b:
                assert fold_prompts[a].isdisjoint(fold_prompts[b])
    decisions = []
    for arm in plan['arms']:
        rows = [r for r in comparisons if r['arm'] == arm]
        raw = float(np.mean([r['raw_gain'] for r in rows]))
        blend = float(np.mean([r['blend_gain'] for r in rows]))
        passed = raw >= 0.01 and blend >= 0.01 and all(r['raw_gain'] > 0 and r['blend_gain'] > 0 for r in rows)
        decisions.append(dict(arm=arm, raw_pooled_gain=raw, blend_pooled_gain=blend, passed=passed))
    return dict(experiment='V46', comparisons=comparisons, decisions=decisions,
                candidate_sha256=hashes, public_score_claim=None,
                scope='Previously inspected prompt-held-out screen; V34 raw component is not the full .622 ensemble.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--reference-project', required=True)
    a = p.parse_args()
    output = Path(a.output)
    destination = output/'screen.json'
    assert not destination.exists(), 'Do not overwrite completed evaluation'
    result = evaluate(json.loads(Path(a.plan).read_text()), output, Path(a.reference_project))
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
