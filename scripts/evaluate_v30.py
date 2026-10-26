"""Independent artifact checks and frozen likelihood pilot promotion gates."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from safetensors.numpy import load_file


def main():
    root=Path('artifacts/improvement-v30');out=root/'kaggle-output'
    plan=json.loads((root/'frozen-plan.json').read_text())
    remote=json.loads((out/'plan.json').read_text())
    for key,value in plan.items():
        if key!='notebook_sha256':assert remote[key]==value,key
    for name,sha in plan['source_sha256'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
    train=pd.read_parquet(out/'training_frame.parquet');valid=pd.read_parquet(out/'validation_frame.parquet')
    for name in ['training_frame.parquet','validation_frame.parquet']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/name),
             pd.read_parquet(Path('artifacts/improvement-v22/kaggle-output')/name),check_dtype=False)
    assert len(train)==20159 and len(valid)==540
    assert set(train.loc[train.stage=='host','group']).isdisjoint(valid.group)
    labels=np.asarray(remote['labels']);assert labels.tolist()==sorted(valid.model.unique())
    ids=valid.id.astype(str).to_numpy();truth=valid.model.to_numpy()
    checkpoints={name:load_file(out/(name+'-adapter.safetensors')) for name in ['initial','external-only','final']}
    for name,tensors in checkpoints.items():
        assert len(tensors)==225 and all(np.isfinite(v).all() for v in tensors.values()),name
        assert set(tensors)==set(checkpoints['initial'])
    for a,b in [('initial','external-only'),('external-only','final')]:
        assert all(not np.array_equal(checkpoints[a][k],v) for k,v in checkpoints[b].items())
    expected=[('pool',i) for i in range(3)]+[('host',i) for i in range(2)]
    parts=[]
    for rank in range(2):
        log=[json.loads(s) for s in (out/f'progress-rank{rank}.jsonl').read_text().splitlines()]
        epochs=[e for e in log if e['event']=='epoch_complete']
        assert [(e['stage'],e['epoch']) for e in epochs]==expected
        assert all(np.isfinite(e['mean_loss']) and e['rows']==(18000 if e['stage']=='pool' else 2159) for e in epochs)
        assert log[-1]['event']=='complete' and log[-1]['optimizer_steps']==1825
        assert log[-1]['skipped_steps']<=18
        d=np.load(out/f'validation-rank{rank}.npz',allow_pickle=False)
        assert np.array_equal(d['labels'],labels) and np.array_equal(d['ids'],ids[rank::2])
        assert d['clipped'].shape==(270,18) and d['mean'].shape==(270,18)
        assert np.isfinite(d['clipped']).all() and np.isfinite(d['mean']).all()
        assert (d['clipped']>=0).all() and (d['clipped']<=1+1e-6).all()
        parts.append({k:d[k].copy() for k in d.files})
    clipped=np.empty((540,18));mean=np.empty_like(clipped)
    for rank,d in enumerate(parts):clipped[rank::2]=d['clipped'];mean[rank::2]=d['mean']
    sparse=np.load('artifacts/improvement-v17/uniform_pool_pilot.npz',allow_pickle=False)
    prior=np.load('artifacts/improvement-v22/kaggle-output/supcon_final_validation.npz',allow_pickle=False)
    for d in [sparse,prior]:assert np.array_equal(d['ids'],ids) and np.array_equal(d['labels'],labels)
    scores=np.log(np.clip(clipped,1e-8,1))
    combined=sparse['scores']+plan['fixed_blend_weight']*scores
    baseline=sparse['scores']+.15*prior['scores']
    good=labels[combined.argmax(1)]==truth;old=labels[baseline.argmax(1)]==truth
    delta=good.astype(float)-old.astype(float)
    groups=valid.group.to_numpy();unique=np.unique(groups)
    assert len(unique)==30 and all((groups==g).sum()==18 for g in unique)
    group_means=np.array([delta[groups==g].mean() for g in unique])
    gate=plan['promotion'];rng=np.random.default_rng(gate['bootstrap_seed'])
    draws=rng.integers(0,len(unique),(gate['bootstrap_samples'],len(unique)))
    interval=np.quantile(group_means[draws].mean(1),[.025,.975])
    result={'verified':True,'clipped_accuracy':float((labels[clipped.argmax(1)]==truth).mean()),
            'mean_likelihood_diagnostic_accuracy':float((labels[mean.argmax(1)]==truth).mean()),
            'fixed_sparse_blend':float(good.mean()),'reference_sparse_blend':float(old.mean()),
            'gain':float(delta.mean()),'prompt_bootstrap_95_interval':interval.tolist(),
            'corrected':int((delta>0).sum()),'newly_wrong':int((delta<0).sum()),
            'qualifies_for_broader_validation':bool(good.mean()>=gate['fixed_sparse_blend_minimum'] and interval[0]>0),
            'submission_authorized':False,'limitations':plan['limitations'],
            'checkpoint_sha256':{name:hashlib.sha256((out/(name+'-adapter.safetensors')).read_bytes()).hexdigest() for name in checkpoints}}
    np.savez_compressed(root/'validation.npz',ids=ids,labels=labels,scores=scores,clipped=clipped,mean=mean)
    (root/'evaluation-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
