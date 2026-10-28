"""One fixed blend screen on aligned, hashed predictions. No parameter search."""
import hashlib,json
from pathlib import Path
import numpy as np


def softmax(x):
    assert np.isfinite(x).all()
    x=np.exp(x-x.max(axis=1,keepdims=True))
    return x/x.sum(axis=1,keepdims=True)


def balanced(y,p,n):
    assert np.array_equal(np.unique(y),np.arange(n))
    return float(np.mean([np.mean(p[y==i]==i) for i in range(n)]))


def main():
    root=Path(__file__).resolve().parents[1]
    plan=json.loads((root/'configs/name-that-llm/v43.json').read_text())
    for name,digest in plan['source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    rows=[]
    for fold in plan['folds']:
        base=np.load(root/f'artifacts/improvement-v34/kaggle-output/8b_ce/fold-{fold}/ce_final_validation.npz',allow_pickle=False)
        bp=softmax(base['scores'])
        for arm in plan['arms']:
            c=np.load(root/f'artifacts/improvement-v40/final-validation/{arm}/fold-{fold}/validation.npz',allow_pickle=False)
            assert all(np.array_equal(base[k],c[k]) for k in ['ids','labels'])
            y=c['targets'];cp=softmax(c['logits'].astype(np.float64))
            p=plan['base_weight']*bp+plan['candidate_weight']*cp
            original=balanced(y,bp.argmax(1),len(c['labels']))
            blended=balanced(y,p.argmax(1),len(c['labels']))
            rows.append(dict(arm=arm,fold=fold,base=original,blend=blended,gain=blended-original,changed=int((p.argmax(1)!=bp.argmax(1)).sum())))
    decisions=[]
    for arm in plan['arms']:
        selected=[r for r in rows if r['arm']==arm]
        gain=float(np.mean([r['gain'] for r in selected]))
        decisions.append(dict(arm=arm,pooled_gain=gain,passed=bool(gain>=plan['gate']['pooled_gain_min'] and all(r['gain']>0 for r in selected))))
    report=dict(experiment='V43',plan_sha256=hashlib.sha256((root/'configs/name-that-llm/v43.json').read_bytes()).hexdigest(),comparisons=rows,decisions=decisions,public_score_claim=None,scope=plan['limitations'])
    (root/'configs/name-that-llm/v43-result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
