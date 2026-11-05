"""Read-only independent verification of downloaded credited public + CE8B outputs.

No polling, model execution, submission, or modification of downloaded artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

CORE={'tfidf','svc4','distort','q14v5','q8v5','ll8v5','n14b','q25_14','deb'}
COMP={'stylegram','stylometry_lgb','ultrabert','debv3l','electl','desk','lora24_seed7','lora24_claude','lora20_q25_instyle','tokattn_deb'}
POOL_SHA='38e48b40e1db0ab1bf6e4f29d3697e37307c49d1ae5020a7af3cd59c484657d5'


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(1048576):h.update(b)
    return h.hexdigest()


def matrix(a,shape,name,prob=False):
    require(a.shape==shape,f'{name}: shape {a.shape} != {shape}')
    require(np.issubdtype(a.dtype,np.number) and np.isfinite(a).all(),f'{name}: nonfinite/nonnumeric')
    if prob:
        require((a>=0).all() and (a<=1+1e-7).all(),f'{name}: outside probability range')
        require(np.allclose(a.sum(1),1,rtol=0,atol=2e-6),f'{name}: rows do not sum to one')


def close(actual,expected,name):
    require(np.allclose(actual,expected,rtol=2e-6,atol=2e-6),
            f'{name}: numerical disagreement (max {np.max(np.abs(actual-expected))})')


def verify(folder,test_csv,asset_manifest):
    folder=Path(folder); files={};warnings=[]
    def path(name):
        p=folder/name;require(p.is_file(),f'Missing required output: {name}')
        files[name]={'bytes':p.stat().st_size,'sha256':digest(p)};return p
    def js(name):return json.loads(path(name).read_text())
    def npz(name):return np.load(path(name),allow_pickle=False)
    test=pd.read_csv(test_csv,dtype={'id':str});require({'id','text'}<=set(test.columns),'Test schema')
    ids=test.id.to_numpy(dtype=str);require(len(ids)>0 and len(set(ids))==len(ids),'Test IDs empty/duplicated')
    shape=(len(ids),18)
    base=npz('public-control-final.npz');aug=npz('public-control-ce8b.npz');core=npz('public-control-core.npz')
    labels=base['labels'].astype(str)
    require(labels.shape==(18,) and len(set(labels))==18,'Expected18 unique labels')
    trusted=json.loads(Path(asset_manifest).read_text())
    require(set(labels)==set(trusted['labels']),'Label set differs from trusted V39 manifest')
    for name,z in [('baseline',base),('augmented',aug),('core',core)]:
        require(np.array_equal(z['ids'].astype(str),ids),f'{name}: ordered IDs mismatch')
        require(np.array_equal(z['labels'].astype(str),labels),f'{name}: label order mismatch')
    for key in ['pre_postprocessing_logits','final_logits','probabilities']:
        matrix(base[key],shape,'baseline/'+key,prob=key=='probabilities')
    close(base['probabilities'],softmax(base['final_logits'],axis=1),'Baseline softmax')
    require(CORE<=set(core.files),'Missing required core score arrays')
    for key in set(core.files)-{'ids','labels'}:matrix(core[key],shape,'core/'+key)
    completion=js('public-control-completion.json')
    require(set(completion['required_core_members'])==CORE,'Unexpected declared core members')
    require(CORE<=set(completion['completed_core_members']),'Incomplete core members')
    require(set(completion['completed_core_members'])==set(core.files)-{'ids','labels'},'Core audit/arrays disagree')
    for key in ['complementary_stack','sibling_residual','pairwise_verifier']:
        require(completion.get(key) is True,'Completion flag false: '+key)
    require(completion.get('ce8b_v39_included') is False,'Baseline unexpectedly includes CE8B')
    comp_audit=js('public-control-comp-audit.json');meta=js('comp_meta.json')
    require(set(comp_audit['required_members'])==COMP,'Unexpected declared complementary members')
    require(COMP<=set(comp_audit['completed_members']),'Incomplete complementary stack')
    require(meta['ids']==ids.tolist(),'Complementary row order mismatch')
    require(len(meta['labels'])==18 and set(meta['labels'])==set(labels),'Complementary labels mismatch')
    require(meta.get('family_gates_applied') is True,'Family gates not applied')
    matrix(np.load(path('comp_logit.npy'),allow_pickle=False),shape,'complementary logits')
    pair=js('v34_pairwise_audit.json')
    require(str(pair['ab_revision'])=='03','Wrong annotation revision')
    require(pair['test_rows']==len(ids) and pair['official_rows']==2699,'Verifier row audit mismatch')
    require(pair.get('A_oof_prompt_grouped') is True,'Verifier prompt grouping absent')
    matrix(np.load(path('v34_pairwise_probs.npy'),allow_pickle=False),shape,'verifier probabilities',prob=True)
    atlas=js('public-control-atlas-audit.json')
    require(0<=atlas['completed']<=atlas['requested']<=len(ids),'Invalid ATLAS coverage')
    require(atlas['requested']==0 or atlas['completed']>0,'ATLAS requested rows but scored none')
    if atlas['completed']<atlas['requested']:
        warnings.append('ATLAS only partially completed its requested rows; notebook permits budget-limited fallback.')
    audit=js('public-control-input-audit.json');pool=audit['open_pool']
    require(pool['rows']==190588 and pool['bytes']==96388050 and pool['sha256_observed']==POOL_SHA,'Wrong OpenPool snapshot')
    for name,version in [('transformers','5.16.1'),('peft','0.20.0')]:
        require(audit['packages'].get(name)==version,f'Unexpected {name} version')
    ce=js('public-control-ce8b-audit.json')
    require(ce['component_log_weight']==.15 and trusted['weight']==.15,'Wrong CE8B fusion weight')
    require(ce['v39_manifest_sha256']==digest(asset_manifest),'CE8B manifest differs from trusted local manifest')
    helper=next(x for x in trusted['files'] if x['path']=='component_inference.py')
    require(ce['helper_sha256']==helper['sha256'],'CE8B helper hash mismatch')
    require(ce['rows']==len(ids),'CE8B row count mismatch')
    require(ce['baseline_outputs_preserved'] is True and ce['neural_training_at_inference'] is False,'Unexpected CE8B execution mode')
    require(ce['fusion_position']=='after corrected public-control final postprocessing; no extra Sinkhorn','Unexpected fusion position')
    for key in ['baseline_logits','component_logits','component_probs','combined_logits','probabilities']:
        matrix(aug[key],shape,'augmented/'+key,prob=key in ['component_probs','probabilities'])
    close(aug['baseline_logits'],base['final_logits'],'Copied baseline logits')
    cp=softmax(aug['component_logits'],axis=1)
    close(aug['component_probs'],cp,'CE8B softmax')
    combined=base['final_logits']+.15*np.log(np.clip(cp,1e-9,1))
    close(aug['combined_logits'],combined,'Fixed .15 CE8B log vote')
    close(aug['probabilities'],softmax(combined,axis=1),'Augmented softmax')
    changed=int(np.sum(combined.argmax(1)!=base['final_logits'].argmax(1)))
    require(ce['changed_predictions']==changed,'Changed-prediction count mismatch')
    for name,logits in [('submission-public-control.parquet',base['final_logits']),('submission.parquet',combined)]:
        frame=pd.read_parquet(path(name));require(list(frame.columns)==['id','model'],name+': schema mismatch')
        require(frame.id.is_unique and np.array_equal(frame.id.astype(str),ids),name+': ordered IDs mismatch')
        require(np.array_equal(frame.model.astype(str),labels[logits.argmax(1)]),name+': labels do not match saved logits')
    return {'verified':True,'rows':len(ids),'labels':labels.tolist(),'changed_predictions':changed,'atlas':atlas,
            'completed_core_members':completion['completed_core_members'],'completed_complementary_members':comp_audit['completed_members'],
            'input_packages':audit['packages'],'warnings':warnings,'files':files,
            'trusted_input_sha256':{'test_csv':digest(test_csv),'asset_manifest':digest(asset_manifest)},
            'limitations':['Checks saved-output arithmetic and internal audits; does not independently rerun models.',
             'Public weight SHA256 are not present in these output audits; runtime checks mainly ensure paths/shapes/sizes.',
             'Cannot reconstruct every baseline postprocessing stage without additional saved intermediate arrays.',
             'Visible-row completion does not establish hidden-test runtime or leaderboard gain.']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--test-csv',required=True)
    p.add_argument('--asset-manifest',default='artifacts/improvement-v39/dataset/v39-asset-manifest.json')
    a=p.parse_args()
    print(json.dumps(verify(a.output_dir,a.test_csv,a.asset_manifest),indent=2))
