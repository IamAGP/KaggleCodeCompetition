"""Verify downloaded v21 artifacts before the submission builder uses them."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from safetensors.numpy import load_file


def audit():
    root=Path('artifacts/improvement-v21')
    output=root/'kaggle-output'
    plan=json.loads((root/'frozen-plan.json').read_text())
    remote_plan=json.loads((output/'plan.json').read_text())
    for key,value in plan.items():
        if key!='notebook_sha256': assert remote_plan[key]==value,key
    for name,digest in plan['source_sha256'].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest()==digest,name
    training=json.loads((output/'full/training_result.json').read_text())
    readout=json.loads((output/'full/readout_result.json').read_text())
    assert training['completed'] and training['rows']==2699 and training['prompts']==150
    assert training['epochs']==2 and training['changed_adapter_tensors']>0
    assert training['labels']==plan['labels']==readout['labels']
    assert readout['portable_readout_check']=='passed' and readout['fit_rows']==2699
    assert readout['C']==plan['readout']['C'] and readout['layers']==plan['readout']['layers']
    for name,digest in readout['artifacts_sha256'].items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest()==digest,name
    host_path=Path('data/name-that-llm/train.csv')
    assert hashlib.sha256(host_path.read_bytes()).hexdigest()==plan['host_sha256']
    host=pd.read_csv(host_path,dtype={'id':str})
    assert sorted(host.model.unique())==plan['labels']
    progress=[json.loads(line) for line in (output/'full/progress.jsonl').read_text().splitlines()]
    epochs=[p for p in progress if p['event']=='epoch_complete']
    assert [p['epoch'] for p in epochs]==[0,1]
    assert progress[-1]['event']=='complete'
    for record in progress:
        if record['event']=='train': assert record['steps']==675 and np.isfinite(record['mean_loss'])
    adapter=load_file(str(output/'full/adapter/adapter_model.safetensors'))
    assert len(adapter)==144 and all(np.isfinite(v).all() for v in adapter.values())
    assert all(np.any(v) for k,v in adapter.items() if 'lora_B' in k)
    assert readout['artifacts_sha256']['full/adapter/adapter_model.safetensors']!=plan['checkpoint_sha256']['adapter/adapter_model.safetensors']
    features=np.load(output/'full/train_features.npy',allow_pickle=False)
    assert features.shape==(2699,3,2560) and np.isfinite(features).all()
    with np.load(output/'full/readout.npz',allow_pickle=False) as state:
        assert state['labels'].tolist()==plan['labels'] and state['layers'].tolist()==plan['readout']['layers']
        assert state['coef'].shape==(3,18,2560) and state['intercept'].shape==(3,18)
        assert state['mean'].shape==state['scale'].shape==(3,2560)
        assert all(np.isfinite(state[key]).all() for key in ['mean','scale','coef','intercept'])
        assert (state['scale']>0).all()
        np.testing.assert_allclose(state['mean'],features.astype(np.float64).mean(axis=0),atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(state['scale'],features.astype(np.float64).std(axis=0),atol=1e-10,rtol=1e-10)
    result={'verified':True,'training':training,'readout':readout,
            'epoch_losses':[p['mean_loss'] for p in epochs],
            'checks':['frozen plan and source hashes','all-host rows and labels','two completed epochs',
                      'finite changed adapters','feature shape and checksums','portable readout fit statistics'],
            'ensemble_validation_claim':None}
    (root/'training-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':audit()
