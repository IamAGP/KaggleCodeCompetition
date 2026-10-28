"""Reject corrupted evidence and exercise the fixed promotion decision."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_blocks_v42 import evaluate, score


def predictions(correct):
    y = np.tile(np.arange(18), 4)
    pred = (y + 1) % 18
    pred[:correct] = y[:correct]
    return dict(ids=np.array([str(i) for i in range(72)]), targets=y,
                labels=np.array([str(i) for i in range(18)]),
                prompts=np.repeat(np.arange(4),18), logits=np.eye(18)[pred])


@pytest.mark.parametrize('correct,expected', [(36,False),(54,True)])
def test_frozen_gate_with_aligned_evidence(tmp_path, correct, expected):
    plan = dict(folds=[1,2], arms=['dense_ce','blocks_supcon'], inputs={}, gate={
        'blocks_supcon_over_v40_blocks_supcon_pooled_min':.01,
        'blocks_supcon_over_v42_dense_ce_pooled_min':.01, 'action':'confirm'})
    reference = tmp_path/'inputs'/'v40-reference'
    output = tmp_path/'output'
    for fold in plan['folds']:
        relative=f'v40-reference/blocks_supcon/fold-{fold}/validation.npz'
        path=reference.parent/relative
        path.parent.mkdir(parents=True)
        np.savez(path, **predictions(36))
        plan['inputs'][relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    for fold in plan['folds']:
        for arm in plan['arms']:
            root=output/arm/f'fold-{fold}'; root.mkdir(parents=True)
            np.savez(root/'validation.npz',**predictions(correct if arm=='blocks_supcon' else 36))
            (root/'run-config.json').write_text(json.dumps(dict(plan=plan,arm=arm,fold=fold)))
    assert evaluate(plan,output,reference)['frozen_screen_pass'] is expected
    # Same-shaped but different examples must never be compared as paired data.
    bad=predictions(correct);bad['ids']=bad['ids'][::-1]
    np.savez(output/'blocks_supcon'/'fold-1'/'validation.npz',**bad)
    with pytest.raises(AssertionError): evaluate(plan,output,reference)


def test_missing_class_cannot_silently_produce_nan_score():
    z=predictions(36);z['targets'][:]=0
    with pytest.raises(AssertionError,match='target class'):score(z)
