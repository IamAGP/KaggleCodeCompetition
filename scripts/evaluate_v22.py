"""Audit the scaled training pilot and apply its already-frozen criteria."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from safetensors.numpy import load_file


def main():
    root=Path('artifacts/improvement-v22')
    out=root/'kaggle-output'
    plan=json.loads((root/'frozen-plan.json').read_text())
    remote=json.loads((out/'training-plan.json').read_text())
    for key,value in plan.items():
        if key not in ['notebook_sha256','worker_sha256','readout_worker_sha256','packaging_template_sha256']:
            assert remote[key]==value,key
    assert hashlib.sha256((out/'worker.py').read_bytes()).hexdigest()==plan['worker_sha256']
    assert hashlib.sha256((out/'readout_worker.py').read_bytes()).hexdigest()==plan['readout_worker_sha256']
    readout_plan=json.loads((out/'readout-plan.json').read_text())
    for key,value in plan['readout'].items(): assert readout_plan[key]==value,key
    fitting=pd.read_parquet(out/'training_frame.parquet')
    validation=pd.read_parquet(out/'validation_frame.parquet')
    for filename in ['training_frame.parquet','validation_frame.parquet']:
        pd.testing.assert_frame_equal(pd.read_parquet(out/filename),
                                      pd.read_parquet(root/'preflight'/filename),check_dtype=False)
    assert len(fitting.loc[fitting.stage=='pool'])==18000
    assert len(fitting.loc[fitting.stage=='host'])==2159 and len(validation)==540
    assert set(fitting.loc[fitting.stage=='host','group']).isdisjoint(validation.group)
    host=pd.read_csv('data/name-that-llm/train.csv',dtype={'id':str})
    assert hashlib.sha256(Path('data/name-that-llm/train.csv').read_bytes()).hexdigest()==plan['host_sha256']
    ids=host.loc[host.id.isin(validation.id),'id'].to_numpy(dtype=str)
    assert ids.tolist()==validation.id.tolist()
    labels=np.asarray(remote['labels'])
    assert labels.tolist()==sorted(host.model.unique())
    truth=host.set_index('id').loc[ids,'model'].to_numpy()
    sparse_path=Path('artifacts/improvement-v17/uniform_pool_pilot.npz')
    sparse=np.load(sparse_path,allow_pickle=False)
    assert np.array_equal(sparse['ids'],ids) and np.array_equal(sparse['labels'],labels)
    thresholds=[float(s) for s in re.findall(r'>=([0-9.]+)',plan['promotion'])]
    assert len(thresholds)==2
    results={}
    for variant,training_variant in [('ce_final','ce'),('supcon_final','ce_supcon')]:
        checkpoint=out/training_variant/'final/adapter/adapter_model.safetensors'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==readout_plan['adapter_sha256'][variant]
        assert checkpoint.read_bytes()!=(out/training_variant/'external_only/adapter/adapter_model.safetensors').read_bytes()
        adapter=load_file(str(checkpoint))
        assert len(adapter)==144 and all(np.isfinite(v).all() for v in adapter.values())
        data=np.load(out/(variant+'_validation.npz'),allow_pickle=False)
        record=json.loads((out/(variant+'_result.json')).read_text())
        assert np.array_equal(data['ids'],ids) and np.array_equal(data['labels'],labels)
        scores=data['scores']
        assert scores.shape==(540,18) and np.isfinite(scores).all()
        assert data['per_layer_scores'].shape==(3,540,18)
        np.testing.assert_allclose(scores,data['per_layer_scores'].mean(0),atol=1e-10,rtol=1e-10)
        accuracy=float((labels[scores.argmax(1)]==truth).mean())
        assert abs(accuracy-record['accuracy'])<1e-12
        features=out/(variant+'_features.npy')
        assert hashlib.sha256(features.read_bytes()).hexdigest()==record['feature_sha256']
        x=np.load(features,mmap_mode='r',allow_pickle=False)
        assert x.shape==(2699,3,2560) and np.isfinite(x).all()
        del x
        history=[json.loads(line) for line in (out/training_variant/'progress.jsonl').read_text().splitlines()]
        epochs=[(p['stage'],p['epoch']) for p in history if p['event']=='epoch_complete']
        assert epochs==[('pool',0),('host',0),('host',1)]
        assert history[-1]['event']=='complete'
        assert all(np.isfinite(p['mean_loss']) for p in history if p['event']=='epoch_complete')
        # Row constants from log-softmax do not change additive argmax.
        combined=sparse['scores']+0.15*scores
        blend=float((labels[combined.argmax(1)]==truth).mean())
        small_path=Path('artifacts/improvement-v19/kaggle-output')/(variant+'_validation.npz')
        small=np.load(small_path,allow_pickle=False)
        assert np.array_equal(small['ids'],ids) and np.array_equal(small['labels'],labels)
        small_accuracy=float((labels[small['scores'].argmax(1)]==truth).mean())
        small_blend=float((labels[(sparse['scores']+0.15*small['scores']).argmax(1)]==truth).mean())
        results[variant]={'accuracy':accuracy,'small_pool_accuracy':small_accuracy,
                          'accuracy_gain':accuracy-small_accuracy,'fixed_sparse_blend':blend,
                          'small_pool_fixed_sparse_blend':small_blend,'blend_gain':blend-small_blend,
                          'qualifies':accuracy>=thresholds[0] and blend>=thresholds[1],
                          'epoch_losses':[p['mean_loss'] for p in history if p['event']=='epoch_complete'],
                          'training_result':history[-1]['result']}
    qualified=[k for k,v in results.items() if v['qualifies']]
    selected=max(qualified,key=lambda k:results[k]['fixed_sparse_blend']) if qualified else None
    result={'verified':True,'thresholds':{'readout_accuracy':thresholds[0],'fixed_sparse_blend':thresholds[1]},
            'results':results,'qualified':qualified,'selected_for_full_host_training':selected,
            'selection_candidates':2,'submission_authorized':False,
            'limitations':plan['limitations'],
            'comparison_artifact_sha256':{str(sparse_path):hashlib.sha256(sparse_path.read_bytes()).hexdigest()}}
    (root/'evaluation-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
