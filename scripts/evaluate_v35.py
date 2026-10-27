"""Apply frozen V35 screen to aligned, finite held-out predictions."""
import argparse
import math
import tempfile
from safetensors.numpy import load_file
from prepare_likelihood_v33 import prepare
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_v34 import balanced_accuracy, sha


def audit_fold(root, global_plan, arm, fold, frozen, split_kind='primary'):
    assert split_kind in ['primary', 'related']
    spec = global_plan['arms'][arm]
    folder = root/arm/f'fold-{fold}'
    expected = dict(spec['parameters'])
    expected.update(experiment='improvement-v35',pilot_fold=fold,labels=global_plan['labels'],
                    model_dir=spec['model_dir'],checkpoint=spec['checkpoint'],checkpoint_sha256=spec['checkpoint_sha256'],
                    confirmation_split=f'{split_kind}-{fold}',confirmation_fold=fold,
                    host_sha256=global_plan['host_sha256'],**global_plan['fold_sizes'][str(fold)])
    assert json.loads((folder/'training-plan.json').read_text()) == expected
    assert json.loads((folder/'plan.json').read_text()) == expected
    with tempfile.TemporaryDirectory() as temp:
        prepare('data/name-that-llm/train.csv',global_plan['fold_by_prompt'],expected,temp)
        for name in ['training_frame.parquet','validation_frame.parquet']:
            pd.testing.assert_frame_equal(pd.read_parquet(folder/name),pd.read_parquet(Path(temp)/name),check_dtype=False)
    valid = pd.read_parquet(folder/'validation_frame.parquet')
    logs = [json.loads(line) for line in (folder/spec['variant']/'progress.jsonl').read_text().splitlines()]
    assert logs[0]['event']=='external_checkpoint_loaded' and logs[0]['sha256']==spec['checkpoint_sha256']
    epochs = [r for r in logs if r['event']=='epoch_complete']
    assert [(r['stage'],r['epoch']) for r in epochs]==[('host',0),('host',1)]
    assert all(np.isfinite(r['mean_loss']) for r in epochs)
    expected_steps = math.ceil(expected['training_rows']/expected['batch_size'])
    updates = [r for r in logs if r['event']=='train']
    assert all(r['stage']=='host' and r['steps']==expected_steps for r in updates)
    assert logs[-1]['event']=='complete'
    result = json.loads((folder/spec['variant']/'result.json').read_text())
    assert result==logs[-1]['result'] and result['completed'] and result['validation_rows']==len(valid)
    checkpoint = folder/spec['variant']/'final/adapter/adapter_model.safetensors'
    original = Path(f"artifacts/improvement-v{spec['origin_version']}/kaggle-output")/spec['variant']/'external_only'
    for relative,digest in spec['checkpoint_sha256'].items():assert sha(original/relative)==digest
    before = load_file(str(original/'adapter/adapter_model.safetensors'))
    after = load_file(str(checkpoint));assert set(before)==set(after)
    assert len(after)==(504 if arm=='8b_ce' else 144)
    assert all(v.shape==before[k].shape and np.isfinite(v).all() for k,v in after.items())
    changed = sum(not np.array_equal(before[k],v) for k,v in after.items());assert changed>0
    readout = dict(expected);readout.update(spec['readout'])
    # Remote absolute fold directories differ from the local download directory.
    actual_readout = json.loads((folder/'readout-plan.json').read_text())
    readout.update(host_path=global_plan['host_path'],upstream_path=actual_readout['upstream_path'],
                   adapter_sha256={spec['readout_variant']:sha(checkpoint)})
    assert actual_readout==readout
    readout_result = json.loads((folder/(spec['readout_variant']+'_result.json')).read_text())
    feature_path = folder/(spec['readout_variant']+'_features.npy')
    assert sha(feature_path)==readout_result['feature_sha256']
    features = np.load(feature_path,mmap_mode='r',allow_pickle=False)
    assert features.shape==(2699,3,spec['hidden_size']) and np.isfinite(features).all()
    with np.load(folder/(spec['readout_variant']+'_validation.npz'),allow_pickle=False) as data:
        assert np.array_equal(data['ids'],valid.id.to_numpy(dtype=str))
        assert data['labels'].tolist()==global_plan['labels']
        assert data['scores'].shape==(len(valid),18) and np.isfinite(data['scores']).all()
        assert data['per_layer_scores'].shape==(3,len(valid),18) and np.isfinite(data['per_layer_scores']).all()
        np.testing.assert_allclose(data['scores'],data['per_layer_scores'].mean(0),rtol=0,atol=1e-12)
        accuracy=float((data['labels'][data['scores'].argmax(1)]==valid.model).mean())
        assert abs(accuracy-readout_result['accuracy'])<1e-12
        assert readout_result['fit_rows']==expected['training_rows'] and readout_result['validation_rows']==len(valid)
        return data['scores'].copy(), {'arm':arm,'fold':fold,'changed_adapter_tensors':changed,
                                     'skipped_steps':result['skipped_steps'],'checkpoint_sha256':sha(checkpoint),
                                     'readout_accuracy':accuracy}


def audit_global(root, frozen):
    assert sha('configs/name-that-llm/v35.json') == frozen['protocol_sha256']
    assert json.loads(Path('configs/name-that-llm/v35.json').read_text()) == frozen['protocol']
    remote=json.loads((root/'plan.json').read_text())
    for key,value in frozen.items():
        if key in ['notebook_sha256','setup_sha256','arms']:continue
        assert remote[key]==value,key
    for arm,spec in frozen['arms'].items():
        assert {k:v for k,v in remote['arms'][arm].items() if k not in ['model_dir','checkpoint']}==spec
    for name,digest in frozen['source_sha256'].items():assert sha(root/name)==digest,name
    if 'portable_runtime' in remote:
        assert remote['portable_runtime']==frozen['protocol']['runtime']['portable']
        manifest=json.loads(Path('artifacts/improvement-v35/portable/manifest.json').read_text())
        assert sha(root/'run.py')==manifest['run_sha256']
        assert manifest['source_notebook_sha256']==frozen['notebook_sha256']
        versions=json.loads((root/'dependency-provenance.json').read_text())['versions']
        for package,version in {'torch':'2.11.0+cu128','transformers':'5.16.1','peft':'0.20.0'}.items():
            assert versions[package]==version
        assert (root/'pip-freeze.txt').stat().st_size>0
    assert json.loads((root/'summary.json').read_text())['completed']
    return remote


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    root=parser.parse_args().output
    frozen=json.loads(Path('artifacts/improvement-v35/frozen-plan.json').read_text())
    protocol=frozen['protocol']
    remote=audit_global(root,frozen)
    assert sha(protocol['primary_folds']['file'])==protocol['primary_folds']['sha256']
    host=pd.read_csv('data/name-that-llm/train.csv',dtype={'id':str,'prompt_id':str})
    assert sha('data/name-that-llm/train.csv')==frozen['host_sha256']
    labels=np.array(frozen['labels']); lookup=dict(zip(labels,range(len(labels))))
    folds=pd.read_csv(protocol['primary_folds']['file'],dtype={'id':str,'prompt_id':str})
    assert host.id.equals(folds.id) and host.prompt_id.equals(folds.prompt_id)
    mapping={p:{'group':'host:'+p,'fold':int(f)} for p,f in folds.groupby('prompt_id').fold.first().items()}
    assert mapping==frozen['fold_by_prompt']
    audits=[]
    truth=[]; predictions={arm:[] for arm in frozen['arms']}; metrics={arm:{} for arm in frozen['arms']}
    for fold in protocol['primary_folds']['train']:
        selected=host.prompt_id.map(lambda p:frozen['fold_by_prompt'][p]['fold'])==fold
        expected=host.loc[selected]
        y=expected.model.map(lookup).to_numpy();truth.extend(y)
        for arm,spec in frozen['arms'].items():
            scores,audit=audit_fold(root,remote,arm,fold,frozen)
            audits.append(audit)
            prediction=scores.argmax(1)
            predictions[arm].extend(prediction)
            metrics[arm][str(fold)]=balanced_accuracy(y,prediction)
    pooled={arm:balanced_accuracy(np.array(truth),np.array(pred)) for arm,pred in predictions.items()}
    gain=pooled['crossed']-pooled['independent']
    passed=gain>=protocol['screen_gate']['pooled_balanced_accuracy_gain_minimum'] and all(
        metrics['crossed'][str(f)]>metrics['independent'][str(f)] for f in protocol['primary_folds']['train'])
    audit={'pooled':pooled,'folds':metrics,'gain':gain,'screen_pass':bool(passed),
           'verified':True,'artifact_audits':audits,'submission_authorized':False,'scope':'Two-fold standalone readout, not full incumbent ensemble'}
    path=root/'v35-screen.json'
    if path.exists():assert json.loads(path.read_text())==audit
    else:path.write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
