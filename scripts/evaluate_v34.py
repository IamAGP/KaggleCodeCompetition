"""Audit V34 outputs and apply the fixed primary confirmation gates."""
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from safetensors.numpy import load_file
from scipy.special import softmax

from diagnose_balance_transfer import balance
from prepare_likelihood_v33 import prepare


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_accuracy(truth, prediction, classes=18):
    counts = np.bincount(truth, minlength=classes)
    assert (counts > 0).all()
    hits = np.bincount(truth, weights=(truth == prediction), minlength=classes)
    return float(np.mean(hits/counts))


def paired_interval(truth, candidate, comparison, groups, folds, spec, classes=18):
    """Resample complete prompts within each fold, preserving paired predictions."""
    unique = np.unique(groups)
    counts = np.zeros((len(unique),classes))
    differences = np.zeros_like(counts)
    group_folds = []
    for i, group in enumerate(unique):
        mask = groups == group
        fold = np.unique(folds[mask]); assert len(fold)==1
        group_folds.append(fold[0])
        counts[i] = np.bincount(truth[mask],minlength=classes)
        delta = (candidate[mask]==truth[mask]).astype(float)-(comparison[mask]==truth[mask])
        differences[i] = np.bincount(truth[mask],weights=delta,minlength=classes)
    group_folds = np.asarray(group_folds)
    rng = np.random.default_rng(spec['seed'])
    draws = np.zeros(spec['samples'])
    for start in range(0,len(draws),256):
        n = min(256,len(draws)-start)
        total = np.zeros((n,classes)); diff = np.zeros_like(total)
        for fold in np.unique(group_folds):
            positions = np.flatnonzero(group_folds==fold)
            selected = positions[rng.integers(0,len(positions),(n,len(positions)))]
            total += counts[selected].sum(1); diff += differences[selected].sum(1)
        assert (total > 0).all()
        draws[start:start+n] = (diff/total).mean(1)
    return np.quantile(draws,spec['interval']).tolist()


def score_primary(host, folds, labels, sparse, candidate, comparison, protocol):
    truth = np.array([dict(zip(labels,range(len(labels))))[m] for m in host.model])
    outputs = {}; metrics = {}
    for name, neural in [('candidate',candidate),('comparison',comparison)]:
        assert neural.shape == sparse.shape == (len(host),len(labels))
        assert np.isfinite(neural).all() and np.isfinite(sparse).all()
        prediction = np.empty(len(host),dtype=int)
        per_fold = {}
        for fold in np.unique(folds):
            mask = folds == fold
            p = balance(softmax(sparse[mask]+protocol['scoring']['component_weight']*neural[mask],axis=1))
            assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
            prediction[mask] = p.argmax(1)
            per_fold[str(fold)] = balanced_accuracy(truth[mask],prediction[mask],len(labels))
        outputs[name] = prediction
        metrics[name] = {'balanced_accuracy':balanced_accuracy(truth,prediction,len(labels)),
                         'ordinary_accuracy':float((prediction==truth).mean()),'folds':per_fold}
    gain = metrics['candidate']['balanced_accuracy']-metrics['comparison']['balanced_accuracy']
    gate = protocol['primary_gate']
    positive = sum(metrics['candidate']['folds'][str(f)]-metrics['comparison']['folds'][str(f)] > gate['material_tolerance'] for f in np.unique(folds))
    interval = paired_interval(truth,outputs['candidate'],outputs['comparison'],host.prompt_id.to_numpy(),
                               folds,protocol['bootstrap'],len(labels))
    passed = (metrics['candidate']['balanced_accuracy']>=gate['candidate_pooled_balanced_accuracy_minimum']
              and gain>=gate['candidate_minus_comparison_minimum']
              and positive>=gate['materially_positive_gain_folds_minimum'] and interval[0]>0)
    return {'metrics':metrics,'gain':gain,'materially_positive_folds':int(positive),
            'paired_prompt_bootstrap_95_interval':interval,'qualifies_for_related_confirmation':bool(passed),
            'submission_authorized':False}, outputs


def audit_fold(root, global_plan, arm, fold, frozen):
    spec = global_plan['arms'][arm]
    folder = root/arm/f'fold-{fold}'
    expected = dict(spec['parameters'])
    expected.update(experiment='improvement-v34',pilot_fold=fold,labels=global_plan['labels'],
                    model_dir=spec['model_dir'],checkpoint=spec['checkpoint'],checkpoint_sha256=spec['checkpoint_sha256'],
                    confirmation_split=f'primary-{fold}',confirmation_fold=fold,
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


def main():
    root = Path('artifacts/improvement-v34'); out = root/'kaggle-output'
    frozen = json.loads((root/'frozen-plan.json').read_text())
    remote = json.loads((out/'plan.json').read_text())
    inputs = json.loads((root/'evaluation-inputs.json').read_text())
    for path,digest in inputs.items():assert sha(path)==digest,path
    for key,value in frozen.items():
        if key in ['notebook_sha256','setup_sha256','arms']:continue
        assert remote[key]==value,key
    for arm,spec in frozen['arms'].items():
        assert {k:v for k,v in remote['arms'][arm].items() if k not in ['model_dir','checkpoint']}==spec
    for name,digest in frozen['source_sha256'].items():assert sha(out/name)==digest
    assert json.loads((out/'summary.json').read_text())['completed']
    host = pd.read_csv('data/name-that-llm/train.csv',dtype={'id':str,'prompt_id':str})
    fold_frame = pd.read_csv(frozen['protocol']['primary_folds']['file'],dtype={'id':str,'prompt_id':str})
    assert host.id.equals(fold_frame.id) and host.prompt_id.equals(fold_frame.prompt_id)
    folds = fold_frame.fold.to_numpy(); labels = np.asarray(frozen['labels'])
    assert labels.tolist()==sorted(host.model.unique())
    sparse = np.load('artifacts/improvement-v32/oof.npz',allow_pickle=False)
    assert np.array_equal(sparse['ids'],host.id) and np.array_equal(sparse['labels'],labels)
    assert np.array_equal(sparse['fold'],folds)
    all_scores = {}; audits = []
    for arm,spec in frozen['arms'].items():
        scores=np.full((len(host),18),np.nan)
        pilot_key='candidate' if arm=='8b_ce' else 'comparison'
        pilot=np.load(frozen['protocol'][pilot_key]['pilot_scores'],allow_pickle=False)
        assert np.array_equal(pilot['ids'],host.loc[folds==0,'id']) and np.array_equal(pilot['labels'],labels)
        scores[folds==0]=pilot['scores']
        completion=json.loads((out/(arm+'-complete.json')).read_text())
        assert completion['completed'] and [r['fold'] for r in completion['folds']]==[1,2,3,4]
        for fold in [1,2,3,4]:
            part,audit=audit_fold(out,remote,arm,fold,frozen)
            scores[folds==fold]=part;audits.append(audit)
        assert np.isfinite(scores).all();all_scores[arm]=scores
    result,predictions=score_primary(host,folds,labels,sparse['scores'],all_scores['8b_ce'],
                                     all_scores['4b_small_supcon'],frozen['protocol'])
    result.update(verified=True,artifact_audits=audits,limitations=frozen['protocol']['limitations'],
                  protocol_sha256=frozen['protocol_sha256'],pilot_fold_reused=True)
    np.savez_compressed(root/'primary-oof.npz',ids=host.id.to_numpy(dtype=str),labels=labels.astype(str),fold=folds,
                        candidate_scores=all_scores['8b_ce'],comparison_scores=all_scores['4b_small_supcon'],
                        candidate_predictions=predictions['candidate'],comparison_predictions=predictions['comparison'])
    (root/'primary-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
