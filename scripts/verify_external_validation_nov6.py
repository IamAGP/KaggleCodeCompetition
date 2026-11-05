"""Independently verify downloaded Nov6 external evaluation outputs; no inference/polling.

Usage: python3 scripts/verify_external_validation_nov6.py --output-dir PATH
Optional --report PATH writes the verified audit, without overwriting a differing file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax

MANIFEST_SHA = '6d7f2cee9c6ec5a027b17f87ddd3f75a4d93f6b46d1d17d9a94e09f380df1443'
PROTOCOL_SHA = '6f831e64e06a2b2d1181f4d6748417d1c7e767d340c6a76cb0aba8e87cf901d7'
LABELS = ['gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra',
 'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash',
 'claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5','claude-fable-5',
 'qwen3.8-max','glm-5.3','glm-5.3-flash','kimi-k3']
KINDS = ['baseline_raw','candidate_raw','baseline_balanced','candidate_balanced']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def close(actual, expected, name):
    require(np.allclose(actual, expected, rtol=0, atol=1e-12), f'{name}: mismatch {actual!r} versus {expected!r}')

def balance(probability):
    # Independently reconstruct the fixed 300 rounds; do not import notebook code.
    factors = np.ones(probability.shape[1], dtype=np.float64)
    target = len(probability) / probability.shape[1]
    for _ in range(300):
        weighted = probability * factors
        normalized = weighted / weighted.sum(axis=1, keepdims=True)
        factors *= target / normalized.sum(axis=0)
    weighted = probability * factors
    return weighted / weighted.sum(axis=1, keepdims=True)

def metrics(y, prediction):
    classes = sorted(set(y))
    correct = prediction == y
    per_class = {label: int(correct[y == label].sum()) for label in classes}
    return {'accuracy':float(correct.mean()),
      'balanced_accuracy_17_observed_classes':float(np.mean([per_class[c]/int((y==c).sum()) for c in classes])),
      'correct':int(correct.sum()),'predicted_missing_gemini_3_7':int((prediction=='gemini-3.7-flash').sum()),
      'per_class_correct':per_class}

def bootstrap(y, baseline, candidate, prompt):
    # Encode cluster/class counts with bincount rather than notebook's nested masks.
    clusters, group_index = np.unique(prompt, return_inverse=True)
    classes, class_index = np.unique(y, return_inverse=True)
    cells = group_index*len(classes)+class_index
    shape = (len(clusters),len(classes))
    counts = np.bincount(cells, minlength=np.prod(shape)).reshape(shape)
    change = (candidate==y).astype(int)-(baseline==y).astype(int)
    gains = np.bincount(cells,weights=change,minlength=np.prod(shape)).reshape(shape)
    rng = np.random.default_rng(61106)
    samples=[];invalid=0
    for _ in range(5000):
        draw=rng.integers(0,len(clusters),size=len(clusters))
        weights=np.bincount(draw,minlength=len(clusters))
        denominator=weights@counts
        if np.any(denominator==0):
            invalid+=1
            continue
        samples.append(float(np.mean((weights@gains)/denominator)))
    require(len(samples)>=4500,'Too few valid cluster bootstrap samples')
    return np.quantile(samples,[.025,.975]).tolist(), {'draws':5000,'valid':len(samples),'invalid_missing_class':invalid,'seed':61106}

def verify(output_dir, cohort_dir, protocol_path):
    output_dir=Path(output_dir);cohort_dir=Path(cohort_dir);protocol_path=Path(protocol_path)
    manifest_path=cohort_dir/'manifest.json'
    require(sha(manifest_path)==MANIFEST_SHA,'Frozen cohort manifest changed')
    require(sha(protocol_path)==PROTOCOL_SHA,'Frozen evaluation protocol changed')
    manifest=json.loads(manifest_path.read_text());protocol=json.loads(protocol_path.read_text())
    require(protocol['cohort_manifest_sha256']==MANIFEST_SHA and protocol['no_tuning'],'Protocol mismatch')
    for name in ['external_test.csv','external_labels.csv']:
        path=cohort_dir/name;entry=manifest['files'][name]
        require(path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],f'Cohort file mismatch: {name}')
    test=pd.read_csv(cohort_dir/'external_test.csv',dtype=str,keep_default_na=False)
    truth=pd.read_csv(cohort_dir/'external_labels.csv',dtype=str,keep_default_na=False)
    require(test.columns.tolist()==['id','text'],'Test schema')
    require(truth.columns.tolist()==['id','model','prompt_id'],'Label schema')
    require(test.id.tolist()==truth.id.tolist() and test.id.is_unique and len(test)==544,'Cohort ID mismatch')
    require(test.text.str.len().gt(0).all() and truth.prompt_id.str.len().gt(0).all(),'Empty text/group')
    require(set(truth.model)==set(LABELS)-{'gemini-3.7-flash'},'Cohort classes')
    require(truth.model.value_counts().eq(32).all(),'Unbalanced frozen cohort')
    files=['v39_scores.npz','external_predictions.parquet','external_comparison.parquet',
           'external-validation-report.json','external-runtime-audit.json']
    require(not (output_dir/'submission.parquet').exists(),'Unexpected competition submission artifact')
    for name in files:require((output_dir/name).is_file(),f'Missing output: {name}')
    with np.load(output_dir/'v39_scores.npz',allow_pickle=False) as z:
        require(set(z.files)=={'ids','labels','baseline_scores','component_scores','combined_scores'},'NPZ schema')
        require(z['ids'].tolist()==test.id.tolist(),'Score IDs/order')
        require(z['labels'].tolist()==LABELS,'Score class order')
        base=z['baseline_scores'].copy();component=z['component_scores'].copy();combined=z['combined_scores'].copy()
    for name,values in [('baseline',base),('component',component),('combined',combined)]:
        require(values.shape==(544,18) and np.isfinite(values).all(),f'Invalid {name} score matrix')
    # Reconstruct the exact fixed .15 log-probability vote, including clipping.
    reconstructed=base+0.15*np.log(np.clip(softmax(component,axis=1),1e-9,1))
    close(combined,reconstructed,'Fixed CE8B addition')
    probabilities={'baseline_raw':softmax(base,axis=1),'candidate_raw':softmax(combined,axis=1)}
    for prefix in ['baseline','candidate']:
        probabilities[prefix+'_balanced']=balance(probabilities[prefix+'_raw'])
    predictions={k:np.asarray(LABELS)[probabilities[k].argmax(axis=1)] for k in KINDS}
    comparison=pd.read_parquet(output_dir/'external_comparison.parquet')
    require(comparison.columns.tolist()==['id']+KINDS,'Comparison schema')
    require(comparison.id.astype(str).tolist()==test.id.tolist() and comparison.id.is_unique,'Comparison IDs')
    for k in KINDS:require(np.array_equal(comparison[k].to_numpy(),predictions[k]),f'Prediction mismatch: {k}')
    prediction=pd.read_parquet(output_dir/'external_predictions.parquet')
    require(prediction.columns.tolist()==['id','model'],'Prediction schema')
    require(prediction.id.astype(str).tolist()==test.id.tolist() and prediction.id.is_unique,'Prediction IDs')
    require(np.array_equal(prediction.model.to_numpy(),predictions['candidate_balanced']),'Final balanced prediction mismatch')
    y=truth.model.to_numpy();b=predictions['baseline_raw'];c=predictions['candidate_raw']
    calculated={k:metrics(y,predictions[k]) for k in KINDS}
    interval,boot=bootstrap(y,b,c,truth.prompt_id.to_numpy())
    delta=calculated['candidate_raw']['balanced_accuracy_17_observed_classes']-calculated['baseline_raw']['balanced_accuracy_17_observed_classes']
    corrections=int(((b!=y)&(c==y)).sum());regressions=int(((b==y)&(c!=y)).sum())
    report=json.loads((output_dir/'external-validation-report.json').read_text())
    expected={'experiment':'nov6-external-validation','manifest_sha256':MANIFEST_SHA,
      'evaluation_protocol_sha256':PROTOCOL_SHA,'rows':544,'observed_classes':17,'score_columns':18,
      'prompt_clusters':truth.prompt_id.nunique(),'bootstrap':boot,'corrections':corrections,'regressions':regressions,
      'submission_performed':False,'public_score_claim':None,
      'primary':'raw full-ensemble score argmax before Sinkhorn; original label-free centered twins retained',
      'secondary':'original 18-class Sinkhorn stress check; unrepresentative class distribution'}
    for key,value in expected.items():require(report.get(key)==value,f'Report mismatch: {key}')
    require(set(report['metrics'])==set(KINDS),'Metric schema')
    for name,values in calculated.items():
        require(set(report['metrics'][name])==set(values),f'Metric fields: {name}')
        for key,value in values.items():
            if isinstance(value,float):close(report['metrics'][name][key],value,f'{name}/{key}')
            else:require(report['metrics'][name][key]==value,f'Metric mismatch: {name}/{key}')
    close(report['primary_delta_balanced_accuracy'],delta,'Primary gain')
    close(report['prompt_cluster_bootstrap_95pct'],interval,'Bootstrap interval')
    audit=json.loads((output_dir/'external-runtime-audit.json').read_text())
    require(audit['rows']==544 and audit['component_weight']==.15 and audit['neural_training_at_inference'] is False,'Runtime audit')
    members={row['member']:row for row in audit['members']}
    required={'svc':1.,'lora_qwen3_14b':.4,'lora_qwen25_14b':.2,'deberta':.25,'ce8b_v39':.15}
    require(set(required)<=set(members) and len(members)==len(audit['members']),'Member inventory')
    require(set(members)<=set(required)|{'lora_qwen3_14b twin','lora_qwen25_14b twin'},'Unexpected member')
    for name,weight in required.items():require(members[name]['weight']==weight and members[name]['rows']==544,f'Member mismatch: {name}')
    for name in set(members)-set(required):require(members[name]['weight']==.1 and 0<members[name]['rows']<=544,f'Twin mismatch: {name}')
    sources=[manifest_path,protocol_path,Path(__file__)]+[cohort_dir/n for n in ['external_test.csv','external_labels.csv']]+[output_dir/n for n in files]
    return {'verified':True,'experiment':'nov6-external-validation','metrics':calculated,
      'primary_delta_balanced_accuracy':delta,'prompt_cluster_bootstrap_95pct':interval,'bootstrap':boot,
      'corrections':corrections,'regressions':regressions,'prompt_clusters':int(truth.prompt_id.nunique()),
      'source_sha256':{str(p):sha(p) for p in sources},
      'scope':'Independent recalculation from saved scores; does not prove remote model execution, external-data provenance, or public leaderboard gain.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--cohort-dir',default='artifacts/nov6-external-holdout-audit/frozen-544')
    parser.add_argument('--protocol',default='notebooks/nov6-external-validation/evaluation-protocol.json')
    parser.add_argument('--report')
    args=parser.parse_args();result=verify(args.output_dir,args.cohort_dir,args.protocol)
    if args.report:
        dest=Path(args.report)
        if dest.exists():require(json.loads(dest.read_text())==result,'Refusing to overwrite a differing verification')
        else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
