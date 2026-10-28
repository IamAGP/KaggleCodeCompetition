"""Apply the unchanged fixed-blend protocol to V42, with exact paired audits."""
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate_v43 import balanced, softmax


def main():
    root = Path(__file__).resolve().parents[1]
    plan_path = root / 'configs/name-that-llm/v44.json'
    plan = json.loads(plan_path.read_text())
    v42_path = root / 'configs/name-that-llm/v42.json'
    assert hashlib.sha256(v42_path.read_bytes()).hexdigest() == plan['v42_plan_sha256']
    v42 = json.loads(v42_path.read_text())
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    comparisons, candidate_hashes = [], {}
    for fold in plan['folds']:
        base = np.load(root / f'artifacts/improvement-v34/kaggle-output/8b_ce/fold-{fold}/ce_final_validation.npz', allow_pickle=False)
        ref = np.load(root / plan['reference_root'] / 'dense_ce' / f'fold-{fold}' / 'validation.npz', allow_pickle=False)
        assert all(np.array_equal(base[k], ref[k]) for k in ['ids', 'labels'])
        assert len(set(base['ids'])) == len(base['ids'])
        bp = softmax(base['scores'])
        for arm in plan['arms']:
            folder = root / plan['candidate_root'] / arm / f'fold-{fold}'
            assert json.loads((folder / 'run-config.json').read_text()) == dict(plan=v42, arm=arm, fold=fold)
            path = folder / 'validation.npz'
            candidate_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            c = np.load(path, allow_pickle=False)
            assert all(np.array_equal(c[k], ref[k]) for k in ['ids', 'labels', 'targets', 'prompts'])
            assert c['logits'].shape == bp.shape
            cp = softmax(c['logits'].astype(np.float64))
            blended = plan['base_weight'] * bp + plan['candidate_weight'] * cp
            y = c['targets']
            before = balanced(y, bp.argmax(1), len(c['labels']))
            after = balanced(y, blended.argmax(1), len(c['labels']))
            comparisons.append(dict(arm=arm, fold=fold, base=before, blend=after, gain=after-before,
                corrected=int(((bp.argmax(1) != y) & (blended.argmax(1) == y)).sum()),
                regressed=int(((bp.argmax(1) == y) & (blended.argmax(1) != y)).sum())))
    decisions = []
    for arm in plan['arms']:
        rows = [r for r in comparisons if r['arm'] == arm]
        gain = float(np.mean([r['gain'] for r in rows]))
        decisions.append(dict(arm=arm, pooled_gain=gain, passed=bool(
            gain >= plan['gate']['pooled_gain_min'] and all(r['gain'] > 0 for r in rows))))
    report = dict(experiment='V44', plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        candidate_sha256=candidate_hashes, comparisons=comparisons, decisions=decisions,
        public_score_claim=None, scope=plan['limitations'])
    (root / 'configs/name-that-llm/v44-result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
