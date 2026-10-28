"""Recompute V42 scores and frozen gates against ID-aligned V40 predictions."""
import argparse
import json
from pathlib import Path
import numpy as np


def score(z):
    assert np.isfinite(z['logits']).all()
    assert len(set(z['ids'])) == len(z['ids'])
    assert z['logits'].shape == (len(z['ids']), len(z['labels']))
    y, pred = z['targets'], z['logits'].argmax(1)
    assert y.shape == (len(z['ids']),)
    assert len(set(z['labels'])) == len(z['labels'])
    assert np.array_equal(np.unique(y), np.arange(len(z['labels']))), 'Missing or invalid target class'
    return float(np.mean([np.mean(pred[y == k] == k) for k in range(len(z['labels']))]))


def evaluate(plan, output, reference):
    scores, parent, anchors = {}, {}, {}
    for fold in plan['folds']:
        relative = f'v40-reference/blocks_supcon/fold-{fold}/validation.npz'
        import hashlib
        assert hashlib.sha256((reference.parent/relative).read_bytes()).hexdigest() == plan['inputs'][relative], 'Reference checksum mismatch'
        ref = np.load(reference/'blocks_supcon'/f'fold-{fold}'/'validation.npz', allow_pickle=False)
        parent[fold] = score(ref)
        anchors[fold] = ref
        for arm in plan['arms']:
            root = output/arm/f'fold-{fold}'
            z = np.load(root/'validation.npz', allow_pickle=False)
            assert all(np.array_equal(z[k], ref[k]) for k in ['ids','targets','labels','prompts'])
            config = json.loads((root/'run-config.json').read_text())
            assert config == dict(plan=plan, arm=arm, fold=fold)
            scores[arm, fold] = score(z)
    pooled = {a:float(np.mean([scores[a,f] for f in plan['folds']])) for a in plan['arms']}
    gains_parent = [scores['blocks_supcon',f]-parent[f] for f in plan['folds']]
    gains_dense = [scores['blocks_supcon',f]-scores['dense_ce',f] for f in plan['folds']]
    gate = plan['gate']
    passed = bool(np.mean(gains_parent) >= gate['blocks_supcon_over_v40_blocks_supcon_pooled_min']
                  and np.mean(gains_dense) >= gate['blocks_supcon_over_v42_dense_ce_pooled_min']
                  and min(gains_parent+gains_dense) > 0)
    result = dict(experiment='V42', pooled=pooled, gains_over_v40=gains_parent,
                  gains_over_dense_ce=gains_dense, frozen_screen_pass=passed,
                  action=gate['action'], public_score_claim=None)
    (output/'screen.json').write_text(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for key in ['plan','output','reference-root']:p.add_argument('--'+key,required=True)
    a=p.parse_args()
    print(json.dumps(evaluate(json.loads(Path(a.plan).read_text()),Path(a.output),Path(a.reference_root)),indent=2))
