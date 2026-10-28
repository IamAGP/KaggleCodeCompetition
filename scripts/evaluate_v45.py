"""Fixed, credited equal-mass normalization; separate from raw V42 evaluation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnose_balance_transfer import balance
from evaluate_v43 import balanced, softmax


def normalize(p):
    q = balance(p)
    assert np.isfinite(q).all() and np.all(q >= 0)
    assert np.allclose(q.sum(1), 1)
    assert np.allclose(q.sum(0), len(q)/q.shape[1], atol=1e-7)
    return q


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'configs/name-that-llm/v45.json'
    plan = json.loads(source.read_text())
    output = root / 'configs/name-that-llm/v45-result.json'
    assert not output.exists(), 'Never overwrite a completed experiment'
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    rows = []
    for fold in plan['folds']:
        base = np.load(root/f'artifacts/improvement-v34/kaggle-output/8b_ce/fold-{fold}/ce_final_validation.npz', allow_pickle=False)
        ref = np.load(root/f'artifacts/improvement-v40/final-validation/dense_ce/fold-{fold}/validation.npz', allow_pickle=False)
        assert all(np.array_equal(base[k], ref[k]) for k in ['ids', 'labels'])
        bp = softmax(base['scores'])
        bq = normalize(bp)
        y = ref['targets']
        score = lambda p: balanced(y, p.argmax(1), len(ref['labels']))
        for arm in plan['arms']:
            c = np.load(root/f'artifacts/improvement-v42/final-validation/{arm}/fold-{fold}/validation.npz', allow_pickle=False)
            assert all(np.array_equal(c[k], ref[k]) for k in ['ids', 'labels', 'targets', 'prompts'])
            cp = softmax(c['logits'].astype(np.float64))
            cq = normalize(cp)
            blended = normalize(plan['base_weight']*bp + plan['candidate_weight']*cp)
            before, after = bq.argmax(1), blended.argmax(1)
            rows.append(dict(arm=arm, fold=fold, raw_candidate=score(cp), balanced_candidate=score(cq),
                raw_base=score(bp), balanced_base=score(bq), balanced_blend=score(blended),
                gain=score(blended)-score(bq), corrected=int(((before!=y)&(after==y)).sum()),
                regressed=int(((before==y)&(after!=y)).sum())))
    summaries=[]
    for arm in plan['arms']:
        selected=[r for r in rows if r['arm']==arm]
        gain=float(np.mean([r['gain'] for r in selected]))
        summaries.append(dict(arm=arm, raw_candidate=float(np.mean([r['raw_candidate'] for r in selected])),
            balanced_candidate=float(np.mean([r['balanced_candidate'] for r in selected])),
            pooled_blend_gain=gain, passed=bool(gain>=plan['gate']['pooled_gain_min'] and all(r['gain']>0 for r in selected))))
    result=dict(experiment='V45',plan_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        rows=rows,summary=summaries,credit=plan['credit'],limitations=plan['limitations'],public_score_claim=None)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
