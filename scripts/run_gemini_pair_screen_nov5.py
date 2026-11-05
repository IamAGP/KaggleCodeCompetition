"""One frozen, explicitly post-selection pair-specialist CPU screen."""
from pathlib import Path
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path('artifacts/nov5-gemini-pair-screen')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def metrics(y, base, new, pair):
    return {'rows': len(y), 'base_correct': int((base==y).sum()),
        'new_correct': int((new==y).sum()), 'accuracy_delta': float((new==y).mean()-(base==y).mean()),
        'corrections': int(((base!=y)&(new==y)).sum()),
        'regressions': int(((base==y)&(new!=y)).sum()), 'changed': int((new!=base).sum()),
        'pair_rows': int(np.isin(y,pair).sum()),
        'base_pair_correct': int(((base==y)&np.isin(y,pair)).sum()),
        'new_pair_correct': int(((new==y)&np.isin(y,pair)).sum()),
        'nonpair_correct_unchanged': bool(np.array_equal((base==y)[~np.isin(y,pair)],(new==y)[~np.isin(y,pair)]))}

def main():
    assert not (OUT/'report.json').exists(), 'Do not silently rerun a completed screen'
    plan_path = OUT/'plan.json'
    plan=json.loads(plan_path.read_text())
    feasibility=json.loads(Path(plan['source_feasibility']).read_text())
    manifest={str(plan_path):sha(plan_path),str(Path(__file__)):sha(__file__)}
    for path,digest in feasibility['source_sha256'].items():
        assert sha(path)==digest,path
    host=pd.read_csv('data/name-that-llm/train.csv',dtype={'id':str})
    all_results=[]; pooled=[]
    with threadpool_limits(limits=2):
        for split in [f'primary-{f}' for f in plan['folds']]+['related-0']:
            is_primary=split.startswith('primary')
            fold=int(split.split('-')[1])
            root=Path('artifacts/improvement-v34')
            folder=(root/'kaggle-output' if is_primary else root/'related/kaggle-output')/'8b_ce'/f'fold-{fold}'
            tr=pd.read_parquet(folder/'training_frame.parquet');va=pd.read_parquet(folder/'validation_frame.parquet')
            assert set(tr.id).isdisjoint(va.id) and set(tr.prompt_id).isdisjoint(va.prompt_id)
            assert set(tr.id)|set(va.id)==set(host.id)
            feature_path=folder/'ce_final_features.npy'
            features=np.load(feature_path,mmap_mode='r')
            fit=host.id.isin(tr.id).to_numpy() & host.model.isin(plan['pair']).to_numpy()
            val=host.id.isin(va.id).to_numpy()
            assert host.loc[val,'id'].tolist()==va.id.tolist()
            scores=[]
            for j in range(3):
                scaler=StandardScaler()
                X=scaler.fit_transform(features[fit,j]);V=scaler.transform(features[val,j])
                clf=LogisticRegression(C=.02,max_iter=1000,random_state=1901)
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    clf.fit(X,host.loc[fit,'model'])
                assert clf.classes_.tolist()==plan['pair']
                scores.append(clf.predict_log_proba(V))
            score=np.mean(scores,axis=0);assert np.isfinite(score).all()
            predfile=root/'primary-oof.npz' if is_primary else root/'related/related-predictions.npz'
            with np.load(predfile,allow_pickle=False) as pred:
                labels=pred['labels'].astype(str)
                baseline=pred['candidate_predictions'] if is_primary else pred['candidate']
                idx=pd.Index(pred['ids']).get_indexer(va.id);assert (idx>=0).all()
                base=labels[baseline[idx]]
            y=va.model.to_numpy();new=base.copy();route=np.isin(base,plan['pair'])
            specialist=np.asarray(plan['pair'])[score.argmax(1)]
            new[route]=specialist[route]
            r=metrics(y,base,new,plan['pair']);r.update(split=split,fit_pair_rows=int(fit.sum()),
                routed_rows=int(route.sum()),binary_pair_accuracy=float((specialist[np.isin(y,plan['pair'])]==y[np.isin(y,plan['pair'])]).mean()))
            all_results.append(r)
            if is_primary:pooled.append((y,base,new))
            np.savez_compressed(OUT/f'{split}.npz',ids=va.id.to_numpy(dtype=str),truth=y.astype(str),baseline=base,candidate=new,specialist=specialist,log_probability=score)
            for p in [feature_path,predfile,folder/'training_frame.parquet',folder/'validation_frame.parquet']:
                manifest[str(p)]=sha(p)
            print(json.dumps(r),flush=True)
    y,b,n=(np.concatenate([p[i] for p in pooled]) for i in range(3))
    result={'experiment':plan['experiment'],'status':'completed_exploratory_only','method_plan_sha256':manifest[str(plan_path)],
        'folds':all_results,'primary_pooled':metrics(y,b,n,plan['pair']),
        'limitations':['Post-selection: pair selected after examining all primary OOF errors.',
        'Related split also previously inspected and overlapping primary population; not independent confirmation.',
        'Baseline is sparse+8B component evaluation, not full public .622 ensemble.',
        'No leaderboard expected gain or automatic promotion claim.'], 'source_sha256':manifest}
    (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['primary_pooled']))

if __name__=='__main__':main()
