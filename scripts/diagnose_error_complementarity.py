"""Post-hoc saved-score audit; no fitting, weight tuning, or gate changes."""
from pathlib import Path
import hashlib
import itertools
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.special import softmax
from diagnose_balance_transfer import balance

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/error-complementarity-20261026'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    started = time.monotonic()
    assert not (OUT/'report.json').exists(), 'Do not overwrite completed diagnostic'
    tracked = {}
    def path(relative):
        p = ROOT/relative
        tracked[relative] = digest(p)
        return p
    protocol = json.loads(path('artifacts/error-complementarity-20261026/protocol.json').read_text())
    host = pd.read_csv(path('data/name-that-llm/train.csv')).set_index('id')
    folds = pd.read_csv(path('artifacts/baseline-v1/folds.csv')).set_index('id')
    valid = pd.read_parquet(path('artifacts/improvement-v18/kaggle-output/validation_frame.parquet'))
    labels = np.sort(valid.model.unique())
    truth = np.searchsorted(labels, valid.model.to_numpy())
    groups = valid.prompt_id.to_numpy()
    unique = np.unique(groups)
    assert len(valid)==540 and len(unique)==30 and len(labels)==18
    assert valid.id.is_unique and (valid.model.value_counts()==30).all()
    assert (folds.loc[valid.id,'fold']==0).all()
    pd.testing.assert_frame_equal(valid[['prompt_id','text','model']].reset_index(drop=True),host.loc[valid.id,['prompt_id','text','model']].reset_index(drop=True))
    checks = {}
    for version in [18,22,25]:
        prefix=f'artifacts/improvement-v{version}/kaggle-output/'
        train=pd.read_parquet(path(prefix+'training_frame.parquet'))
        other=pd.read_parquet(path(prefix+'validation_frame.parquet'))
        pd.testing.assert_frame_equal(valid,other,check_dtype=False)
        fitting=train[train.stage=='host']
        assert len(fitting)==2159
        assert not set(fitting.id)&set(valid.id)
        assert not set(fitting.prompt_id)&set(valid.prompt_id)
        assert (folds.loc[fitting.id,'fold']!=0).all()
        pd.testing.assert_frame_equal(fitting[['prompt_id','text','model']].reset_index(drop=True),host.loc[fitting.id,['prompt_id','text','model']].reset_index(drop=True))
        assert not set(train.text)&set(valid.text)
        checks[str(version)]={'host_fit_rows':len(fitting),'external_rows':len(train)-len(fitting),'validation_rows':len(other),'disjoint_host_prompts':True,'no_exact_validation_text_in_training':True}
    plan=json.loads(path('artifacts/improvement-v19/kaggle-output/plan.json').read_text())
    assert plan['upstream_version']==2 and plan['pilot_fold']==0
    assert plan['host_sha256']==digest(ROOT/'data/name-that-llm/train.csv')
    path('artifacts/improvement-v19/kaggle-output/evaluation-audit.json')
    path('artifacts/improvement-v22/evaluation-audit.json')
    path('artifacts/improvement-v25/evaluation-audit.json')
    files={
        'sparse':'artifacts/improvement-v17/uniform_pool_pilot.npz',
        'v19_4b_small_supcon':'artifacts/improvement-v19/kaggle-output/supcon_final_validation.npz',
        'v22_4b_large_supcon':'artifacts/improvement-v22/kaggle-output/supcon_final_validation.npz',
        'v25_8b_ce':'artifacts/improvement-v25/kaggle-output/ce_final_validation.npz',
        'v25_8b_supcon':'artifacts/improvement-v25/kaggle-output/supcon_final_validation.npz',
    }
    scores={}
    for name,relative in files.items():
        with np.load(path(relative),allow_pickle=False) as saved:
            assert np.array_equal(saved['labels'],labels)
            assert np.array_equal(saved['ids'],valid.id.astype(str))
            s=saved['scores'].copy()
            assert s.shape==(540,18) and np.isfinite(s).all()
            scores[name]=s
        if name!='sparse':
            result=json.loads(path(relative.replace('_validation.npz','_result.json')).read_text())
            assert result['fit_rows']==2159 and result['validation_rows']==540
            assert abs(float((s.argmax(1)==truth).mean())-result['accuracy'])<1e-12
    def metric(pred):
        recalls=[float((pred[truth==i]==i).mean()) for i in range(18)]
        return {'balanced_accuracy':float(np.mean(recalls)),'correct':int((pred==truth).sum()),'per_class_recall':dict(zip(labels,recalls))}
    def confusion(pred):
        matrix=np.zeros((18,18),dtype=int)
        np.add.at(matrix,(truth,pred),1)
        pairs=[]
        for a,b in itertools.combinations(range(18),2):
            n=int(matrix[a,b]+matrix[b,a])
            if n:
                m=((truth==a)&(pred==b))|((truth==b)&(pred==a))
                pairs.append({'models':[str(labels[a]),str(labels[b])],'errors':n,'a_to_b':int(matrix[a,b]),'b_to_a':int(matrix[b,a]),'distinct_prompts':int(len(np.unique(groups[m])))})
        pairs.sort(key=lambda x:(-x['errors'],x['models']))
        nerr=int((pred!=truth).sum())
        return {'matrix':matrix.tolist(),'pairs':pairs,'top3_share_of_errors':sum(x['errors'] for x in pairs[:3])/nerr}
    draws=np.random.default_rng(protocol['bootstrap']['seed']).integers(0,len(unique),(protocol['bootstrap']['draws'],len(unique)))
    results={}; predictions={}
    ref='v19_4b_small_supcon'
    for view in ['standalone','fixed_sparse_blend_balanced']:
        pred={name:(s.argmax(1) if view=='standalone' else balance(softmax(scores['sparse']+.15*s,axis=1)).argmax(1)) for name,s in scores.items() if name!='sparse'}
        predictions[view]=pred
        comparisons={}
        r=pred[ref];rc=r==truth
        for name,p in pred.items():
            if name==ref:continue
            cc=p==truth;delta=cc.astype(int)-rc.astype(int)
            gp=[{'prompt_id':str(g),'reference_correct':int(rc[groups==g].sum()),'candidate_correct':int(cc[groups==g].sum()),'net_correct':int(delta[groups==g].sum())} for g in unique]
            gd=np.array([x['net_correct']/18 for x in gp])
            bothwrong=(~rc)&(~cc)
            comparisons[name]={
                'corrected_reference_errors':int((cc&~rc).sum()),'new_errors':int((rc&~cc).sum()),
                'shared_errors':int(bothwrong.sum()),'shared_errors_same_wrong_label':int((bothwrong&(p==r)).sum()),
                'shared_errors_different_wrong_labels':int((bothwrong&(p!=r)).sum()),
                'prediction_agreement':float((p==r).mean()),'net_accuracy_gain':float(delta.mean()),
                'prompt_bootstrap_95_gain_interval':np.quantile(gd[draws].mean(1),[.025,.975]).tolist(),
                'oracle_either_correct_upper_bound':float((rc|cc).mean()),
                'oracle_note':'Uses truth to pick the correct detector per row; not an achieved classifier or score.',
                'prompt_wins':int((gd>0).sum()),'prompt_ties':int((gd==0).sum()),'prompt_losses':int((gd<0).sum()),
                'per_prompt':gp,
            }
        results[view]={'metrics':{n:metric(p) for n,p in pred.items()},'comparisons_to_reference':comparisons,'confusions':{n:confusion(p) for n,p in pred.items()}}
    # Confusion identity stability across candidates, not independently held-out validation.
    top_sets={n:{tuple(x['models']) for x in confusion(p)['pairs'][:5]} for n,p in predictions['fixed_sparse_blend_balanced'].items()}
    common=sorted(set.intersection(*top_sets.values()))
    report={'created_at':datetime.now(timezone.utc).isoformat(),'protocol':protocol,'provenance_checks':checks,'labels':labels.tolist(),'results':results,'pairs_in_top5_of_all_four':common,'full_incumbent_ensemble_evaluated':False,'limitations':[
        'This verifies saved frame and prediction provenance, not a new independent audit of all historical optimizer steps.',
        'V19 is a held-out analogue of the V21 added component; it is not the full V21 0.622 system.',
        'Existing repeatedly inspected fold; bootstrap intervals do not correct adaptive selection bias.',
        'No trained specialist or learned gating rule was tested. Oracle scores are unattainable without knowing answers.',
        'Equal-class balancing follows existing diagnostic for comparison; hidden test class counts may differ.',
        'External data were previously screened for overlap, but this check only re-verifies exact text disjointness.',
    ],'input_sha256':tracked,'worker_sha256':digest(Path(__file__)),'runtime_seconds':time.monotonic()-started}
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    # Private row evidence, not for public Git.
    rows=valid[['id','prompt_id','model']].copy()
    for view,preds in predictions.items():
        for name,p in preds.items():rows[view+'__'+name]=labels[p]
    rows.to_csv(OUT/'row-predictions.csv',index=False)
    manifest={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in OUT.iterdir() if p.is_file()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'runtime_seconds':report['runtime_seconds'],'results':{v:{'metrics':{n:m['balanced_accuracy'] for n,m in d['metrics'].items()},'comparisons':{n:{k:x for k,x in c.items() if k!='per_prompt'} for n,c in d['comparisons_to_reference'].items()},'top_pairs':d['confusions'][ref]['pairs'][:5],'top3_error_share':d['confusions'][ref]['top3_share_of_errors']} for v,d in results.items()},'common_pairs':common},indent=2))

if __name__=='__main__':main()
