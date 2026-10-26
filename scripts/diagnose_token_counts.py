"""Recorded CPU-only tokenizer-count diagnostic; does not change old experiments."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import hashlib,json,time,platform,sys,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,accuracy_score
from sklearn.exceptions import ConvergenceWarning

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/token-count-pilot-20261026'
MODELS=['Qwen/Qwen3-8B','Qwen/Qwen3.5-4B','Qwen/Qwen3.8-27B','google/gemma-3-12b-it','deepseek-ai/DeepSeek-R1-Distill-Llama-8B']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def log(event,**kw):
    msg=json.dumps({'elapsed_seconds':round(time.perf_counter()-start,3),'event':event,**kw})
    print(msg,flush=True)
    with (OUT/'debug.jsonl').open('a') as f:f.write(msg+'\n')
def main():
    if (OUT/'report.json').exists():raise RuntimeError('Existing result; refuse silent rerun')
    train=ROOT/'data/name-that-llm/train.csv'; folds=ROOT/'artifacts/baseline-v1/folds.csv'
    df=pd.read_csv(train).merge(pd.read_csv(folds),on=['id','prompt_id'],validate='one_to_one')
    assert len(df)==2699 and df.groupby('prompt_id').fold.nunique().max()==1
    labels=sorted(df.model.unique()); y=np.array([labels.index(v) for v in df.model]); texts=df.text.tolist()
    lengths=np.array([[len(t),len(t.split()),len(t.encode('utf8'))] for t in texts],dtype=float)
    logs=np.log1p(lengths)
    length_features=np.column_stack([logs,lengths[:,0]/np.maximum(lengths[:,1],1),lengths[:,2]/np.maximum(lengths[:,0],1),logs**2])
    counts=[]; tokenizer_records=[]
    for name in MODELS:
        paths=sorted((Path.home()/'.cache/huggingface/hub'/('models--'+name.replace('/','--'))).glob('snapshots/*/tokenizer.json'))
        if not paths:raise FileNotFoundError(name)
        path=paths[-1]; tok=Tokenizer.from_file(str(path));tok.no_padding();tok.no_truncation()
        c=np.array([len(e.ids) for e in tok.encode_batch(texts,add_special_tokens=False)])
        assert np.all(c>0)
        counts.append(c); tokenizer_records.append({'model':name,'snapshot':path.parent.name,'sha256':sha(path),'path':str(path),'min_count':int(c.min()),'max_count':int(c.max())})
        log('tokenizer_complete',model=name,mean_count=float(c.mean()))
    counts=np.column_stack(counts);clog=np.log1p(counts)
    combined=np.column_stack([length_features,clog,counts/lengths[:,0,None],counts/lengths[:,1,None],counts/counts[:,0,None]])
    features={'length_only':length_features,'counts_only':clog,'length_plus_counts':combined,'negative_control':combined}
    rng=np.random.default_rng(261026);yshuf=y.copy()
    for _,ids in df.groupby('prompt_id').groups.items():yshuf[np.array(list(ids))]=rng.permutation(y[np.array(list(ids))])
    predictions={};metrics={};convergence=[]
    for arm,X in features.items():
        pred=np.full(len(y),-1,dtype=int);scores=np.zeros((len(y),len(labels)));foldmetrics=[]
        for fold in sorted(df.fold.unique()):
            fit=df.fold.to_numpy()!=fold;valid=~fit
            assert not set(df.loc[fit,'prompt_id']) & set(df.loc[valid,'prompt_id'])
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',ConvergenceWarning)
                model.fit(X[fit],(yshuf if arm=='negative_control' else y)[fit])
            convergence.extend([{'arm':arm,'fold':int(fold),'message':str(w.message)} for w in caught])
            pred[valid]=model.predict(X[valid]);scores[valid]=model.predict_proba(X[valid])
            fm={'fold':int(fold),'rows':int(valid.sum()),'balanced_accuracy':balanced_accuracy_score(y[valid],pred[valid])};foldmetrics.append(fm)
            log('fold_complete',arm=arm,**fm)
        predictions[arm]=pred
        metrics[arm]={'balanced_accuracy':balanced_accuracy_score(y,pred),'accuracy':accuracy_score(y,pred),'folds':foldmetrics}
        np.savez_compressed(OUT/(arm+'_oof.npz'),ids=df.id.to_numpy(dtype=str),labels=np.array(labels),y=y,pred=pred,probabilities=scores,folds=df.fold.to_numpy())
    # Cluster bootstrap over prompts; compute per-model denominators in each draw.
    groups=[np.array(list(ix)) for ix in df.groupby('prompt_id').groups.values()]
    class_n=np.array([np.bincount(y[ix],minlength=len(labels)) for ix in groups])
    correct={arm:np.array([np.bincount(y[ix],weights=(pred[ix]==y[ix]),minlength=len(labels)) for ix in groups]) for arm,pred in predictions.items()}
    rng=np.random.default_rng(261026);draws=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=2000)
    denominators=draws@class_n
    boot={arm:np.mean((draws@v)/denominators,axis=1) for arm,v in correct.items()}
    delta=boot['length_plus_counts']-boot['length_only']
    ref=predictions['length_only']==y;cand=predictions['length_plus_counts']==y
    duplicate_counts=[]
    for i,a in enumerate(MODELS):
        for j in range(i):
            duplicate_counts.append({'a':a,'b':MODELS[j],'fraction_identical_count':float(np.mean(counts[:,i]==counts[:,j]))})
    report={'experiment':'token-count-pilot-20261026','created_utc':pd.Timestamp.now(tz='UTC').isoformat(),'elapsed_seconds':time.perf_counter()-start,'rows':len(df),'prompts':df.prompt_id.nunique(),'classes':labels,'inputs':{'train_sha256':sha(train),'folds_sha256':sha(folds),'protocol_sha256':sha(OUT/'protocol.json'),'script_sha256':sha(Path(__file__))},'tokenizers':tokenizer_records,'metrics':metrics,'combined_minus_length':{'balanced_accuracy':metrics['length_plus_counts']['balanced_accuracy']-metrics['length_only']['balanced_accuracy'],'prompt_bootstrap_95_interval':np.quantile(delta,[.025,.975]).tolist(),'corrected_errors':int(np.sum(cand & ~ref)),'introduced_errors':int(np.sum(ref & ~cand))},'identical_counts':duplicate_counts,'convergence_warnings':convergence,'environment':{'python':sys.version,'platform':platform.platform()},'limitations':['Not the exact 18 author tokenizers; no Claude/OpenAI API tokenizer counts were collected.','Some cached tokenizer families are closely related; distinct filenames need not mean distinct segmentation.','Features encode length and lexical segmentation; incremental value does not prove identity independent of all length effects.','Existing prompt folds and labels have been inspected in prior development. This is screening, not pristine confirmation.','No comparison to or gain over full submitted ensemble established.','No GPU, model weights, network/API calls, or new paid compute used.']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame(counts,columns=MODELS).assign(id=df.id).to_csv(OUT/'counts.csv',index=False)
    log('completed',metrics={k:round(v['balanced_accuracy'],6) for k,v in metrics.items()},gain=report['combined_minus_length'])
    manifest=[{'path':str(p.relative_to(ROOT)),'size':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':
    start=time.perf_counter();main()
