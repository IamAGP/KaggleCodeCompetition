"""Descriptive fixed-weight overlap check; no tuning or release-policy change."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import log_softmax


def main():
    sparse_path=Path('artifacts/improvement-v17/uniform_pool_pilot.npz')
    component_root=Path('artifacts/improvement-v19/kaggle-output')
    plan=json.loads(Path('artifacts/improvement-v21/frozen-plan.json').read_text())
    weight=plan['blend']['new_member_log_probability_weight']
    sparse=np.load(sparse_path,allow_pickle=False)
    host=pd.read_csv('data/name-that-llm/train.csv',dtype={'id':str}).set_index('id').loc[sparse['ids']]
    y=host.model.to_numpy();labels=sparse['labels']
    scores=log_softmax(sparse['scores'],axis=1)
    pred=labels[scores.argmax(1)]
    results={}
    paths=[sparse_path]
    for variant in ['base','ce_final','supcon_final']:
        path=component_root/(variant+'_validation.npz');paths.append(path)
        component=np.load(path,allow_pickle=False)
        assert np.array_equal(component['ids'],sparse['ids'])
        assert np.array_equal(component['labels'],labels)
        extra=log_softmax(component['scores'],axis=1)
        individual=labels[extra.argmax(1)]
        combined=labels[(scores+weight*extra).argmax(1)]
        results[variant]={'component_accuracy':float((individual==y).mean()),
                          'combined_accuracy':float((combined==y).mean()),
                          'combined_corrected':int(((pred!=y)&(combined==y)).sum()),
                          'combined_new_errors':int(((pred==y)&(combined!=y)).sum()),
                          'component_correct_where_sparse_wrong':int(((pred!=y)&(individual==y)).sum()),
                          'sparse_correct_where_component_wrong':int(((pred==y)&(individual!=y)).sum())}
    result={'kind':'Descriptive fixed-weight complementarity using saved pilot predictions',
            'rows':len(y),'component_weight':weight,'sparse_accuracy':float((pred==y).mean()),
            'results':results,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'limitations':['V17 capped two-view sparse pilot is not the complete v16 ensemble',
                           'V17 screened normalized exact overlap, unlike v18 near-match screening',
                           'Same repeatedly inspected fold','No weight sweep or release-policy change']}
    Path('artifacts/improvement-v21/sparse-complementarity-diagnostic.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
