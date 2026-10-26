"""Extend the fixed V17 sparse reference to uncomputed folds; no tuning."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning


def main():
    root=Path('artifacts/improvement-v32');root.mkdir(exist_ok=True)
    assert not (root/'result.json').exists(),'Completed reference is frozen'
    plan=json.loads(Path('configs/name-that-llm/v32.json').read_text())
    for name,path in plan['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==plan['input_sha256'][name]
    plan['worker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (root/'frozen-plan.json').write_text(json.dumps(plan,indent=2))
    host=pd.read_csv(plan['sources']['host'],dtype={'id':str,'prompt_id':str})
    folds=pd.read_csv(plan['sources']['folds'],dtype={'id':str,'prompt_id':str})
    assert folds.id.equals(host.id) and folds.prompt_id.equals(host.prompt_id)
    selected=pd.read_csv(plan['sources']['selected_pool_ids'],dtype=str)
    pool=pd.read_parquet(plan['sources']['pool'])
    pool.id=pool.id.astype(str);assert pool.id.is_unique
    pool=pool.set_index('id').loc[selected.id].reset_index()
    assert pool[['id','source','model']].astype(str).equals(selected)
    labels=np.sort(host.model.unique())
    scores=np.full((len(host),18),np.nan);metrics={}
    reused=np.load(plan['sources']['fold0'],allow_pickle=False)
    first=folds.fold.to_numpy()==0
    assert np.array_equal(reused['ids'],host.loc[first,'id']) and np.array_equal(reused['labels'],labels)
    scores[first]=reused['scores']
    metrics['0']={'accuracy':float((labels[scores[first].argmax(1)]==host.loc[first,'model']).mean()),'reused':True}
    for fold in plan['fit_folds']:
        path=root/f'fold-{fold}.npz'
        if path.exists():raise RuntimeError('Partial outputs exist; inspect before resuming')
        begin=time.monotonic();heldout=folds.fold.to_numpy()==fold
        train=host.loc[~heldout];valid=host.loc[heldout]
        assert set(train.prompt_id).isdisjoint(valid.prompt_id)
        fitting=pd.concat([train,pool],ignore_index=True)
        blocks=[];test=[]
        for analyzer,ngram,cap in [('char',(3,5),50000),('word',(1,2),30000)]:
            vectorizer=TfidfVectorizer(analyzer=analyzer,ngram_range=ngram,max_features=cap,
                min_df=2,lowercase=False,sublinear_tf=True,dtype=np.float32)
            blocks.append(vectorizer.fit_transform(fitting.text));test.append(vectorizer.transform(valid.text))
        x=hstack(blocks).tocsr();v=hstack(test).tocsr()
        weights=np.r_[np.ones(len(train)),np.full(len(pool),plan['external_weight'])]
        model=LinearSVC(C=plan['C'],class_weight=plan['class_weight'],max_iter=plan['max_iter'],
                        random_state=plan['seed'],dual='auto')
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning);model.fit(x,fitting.model,sample_weight=weights)
        assert np.array_equal(model.classes_,labels)
        prediction=model.decision_function(v);assert np.isfinite(prediction).all()
        scores[heldout]=prediction
        np.savez_compressed(path,ids=valid.id.to_numpy(dtype=str),labels=labels.astype(str),scores=prediction)
        record={'fold':fold,'accuracy':float((labels[prediction.argmax(1)]==valid.model).mean()),
                'seconds':time.monotonic()-begin,'iterations':int(model.n_iter_),'rows':len(valid)}
        metrics[str(fold)]=record
        with (root/'progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    assert np.isfinite(scores).all()
    np.savez_compressed(root/'oof.npz',ids=host.id.to_numpy(dtype=str),labels=labels.astype(str),scores=scores,fold=folds.fold.to_numpy())
    result={'accuracy':float((labels[scores.argmax(1)]==host.model).mean()),'folds':metrics,
            'submission_authorized':False,'purpose':'Reference only','limitations':plan['limitations']}
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
