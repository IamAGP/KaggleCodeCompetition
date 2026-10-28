"""Descriptive paired error audit; no fitting, weight search or promotion."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd


def run():
    root=Path(__file__).resolve().parents[1]
    rows=[];sources={}
    def load(path):
        sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
        return np.load(path,allow_pickle=False)
    for fold in [1,2]:
        base=root/f'artifacts/improvement-v34/kaggle-output/8b_ce/fold-{fold}'
        z=load(base/'ce_final_validation.npz')
        fit=pd.read_parquet(base/'training_frame.parquet')
        valid=pd.read_parquet(base/'validation_frame.parquet')
        assert set(fit.id).isdisjoint(valid.id)
        assert set(fit.prompt_id).isdisjoint(valid.prompt_id)
        assert valid.id.tolist()==z['ids'].tolist()
        y=valid.model.map({v:i for i,v in enumerate(z['labels'])}).to_numpy()
        assert np.array_equal(y,valid.label.to_numpy())
        b=z['scores'].argmax(1);bc=b==y
        assert np.isfinite(z['scores']).all()
        for arm in ['dense_ce','blocks_ce','dense_supcon','blocks_supcon']:
            p=root/f'artifacts/improvement-v40/final-validation/{arm}/fold-{fold}/validation.npz'
            a=load(p)
            assert all(np.array_equal(a[k],z[k]) for k in ['ids','labels'])
            assert np.array_equal(a['targets'],y)
            assert np.array_equal(a['prompts'],valid.prompt_id.to_numpy())
            ac=a['logits'].argmax(1)==y
            rows.append(dict(fold=fold,arm=arm,rows=len(y),base_correct=int(bc.sum()),
                             candidate_correct=int(ac.sum()),candidate_correct_base_wrong=int((ac&~bc).sum()),
                             base_correct_candidate_wrong=int((bc&~ac).sum()),both_wrong=int((~bc&~ac).sum()),
                             oracle_either_correct=int((bc|ac).sum())))
    result=dict(scope='V40 versus the raw V34 8B readout on previously inspected folds; not the full 0.622 ensemble',
                interpretation='Oracle either-correct uses true labels and is NOT an attainable reported model score. No blend was fitted or evaluated.',
                comparisons=rows,source_sha256=sources)
    out=root/'artifacts/improvement-v42/v40-complementarity.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
