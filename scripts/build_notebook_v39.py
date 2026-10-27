"""Audited full-host V34 CE8B inference addition; no neural fitting at inference."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import nbformat


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inference_source(source):
    """Keep audited feature/predict methods exactly in AST; remove fitting and main."""
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ContrastiveReadout')
    wanted=['__init__','features','predict']
    methods=[n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in wanted]
    assert [n.name for n in methods]==wanted
    cls.body=methods
    tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]+[cls]
    result='"""Inference-only extraction of audited V34 CE8B feature/readout methods."""\n'+ast.unparse(tree)+'\n'
    compiled=ast.parse(result)
    recovered=next(n for n in compiled.body if isinstance(n,ast.ClassDef))
    assert [ast.dump(n,include_attributes=False) for n in recovered.body]==[ast.dump(n,include_attributes=False) for n in methods]
    compile(result,'v39-inference-helper','exec')
    return result


def build():
    full=Path('artifacts/improvement-v34/full-host');output=full/'kaggle-output'
    assert (full/'training-audit.json').is_file(),'Full-host audit not retrieved; V39 remains unbuilt'
    from audit_full_v34 import audit
    audit() # Re-audit real local weights/features and recorded training; no flags-only trust.
    plan=json.loads((full/'frozen-plan.json').read_text())
    approved=json.loads((full/'training-audit.json').read_text())
    assert approved['verified'] and approved['training']['completed']
    assert approved['training']['rows']==2699 and approved['readout']['fit_rows']==2699
    assert approved['readout']['portable_readout_check']=='passed'
    assert plan['related_qualification']['verified'] and plan['related_qualification']['qualifies_for_full_host_training']
    assert plan['training_parameters']['contrastive_weight']==0
    assert plan['readout']['layers']==[12,24,36] and plan['readout']['C']==.02
    original_source=output/'component_v34.py'
    assert sha(original_source)==plan['source_sha256']['component_v34.py']
    helper=inference_source(original_source.read_text())
    baseline_path=Path('notebooks/improvement-v16/name-that-llm-two-stage-v16.ipynb')
    baseline_plan=json.loads(Path('configs/name-that-llm/v16.json').read_text())
    assert sha(baseline_path)==baseline_plan['notebook_sha256']
    nb=nbformat.read(baseline_path,4)
    codes=[c.source for c in nb.cells if c.cell_type=='code'];old_audit=nb.cells[-1].source
    out=Path('artifacts/improvement-v39');folder=Path('notebooks/improvement-v39')
    assert not out.exists() and not folder.exists(),'Never overwrite a built V39'
    dataset=out/'dataset';dataset.mkdir(parents=True)
    files={
        'adapter/adapter_model.safetensors':output/'full/adapter/adapter_model.safetensors',
        'adapter/adapter_config.json':output/'full/adapter/adapter_config.json',
        'readout.npz':output/'full/readout.npz',
        'training-result.json':output/'full/training_result.json',
        'readout-result.json':output/'full/readout_result.json',
        'training-audit.json':full/'training-audit.json',
    }
    for relative,source in files.items():
        target=dataset/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    (dataset/'component_inference.py').write_text(helper)
    manifest={'experiment':'improvement-v39','candidate':'full-host V34 Qwen3-8B CE','neural_training_at_inference':False,
              'model':'qwen-lm/qwen-3/Transformers/8b-base/1','weight':.15,'labels':plan['labels'],
              'layers':plan['readout']['layers'],'max_length':plan['training_parameters']['max_length'],
              'batch_size':plan['readout']['batch_size'],'original_component_sha256':sha(original_source),
              'full_host_plan_sha256':sha(full/'frozen-plan.json'),
              'files':[{'path':str(p.relative_to(dataset)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(dataset.rglob('*')) if p.is_file()],
              'contains_raw_training_text':False,'contains_predictions':False,
              'attribution':'Baseline Nishant Dahal and original contributors. Qwen backbones; established LoRA/QLoRA. Our fixed CE8B candidate qualified under V34 development gates; no public gain claimed.'}
    manifest_path=dataset/'v39-asset-manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2))
    dataset_id='adithyagiri/name-that-llm-v39-private-ce8b'
    (dataset/'dataset-metadata.json').write_text(json.dumps({'id':dataset_id,'title':'Name That LLM V39 Private CE8B Inference Assets','licenses':[{'name':'CC-BY-SA-4.0'}]},indent=2))
    addition=f'''import hashlib,importlib.util
V39_MANIFEST_SHA256={sha(manifest_path)!r}
roots=[p.parent for p in Path('/kaggle/input').rglob('v39-asset-manifest.json') if hashlib.sha256(p.read_bytes()).hexdigest()==V39_MANIFEST_SHA256]
assert len(roots)==1,roots
v39_root=roots[0]
v39_manifest=json.loads((v39_root/'v39-asset-manifest.json').read_text())
for item in v39_manifest['files']:
    path=v39_root/item['path'];assert path.stat().st_size==item['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],item['path']
module_path=v39_root/'component_inference.py'
spec=importlib.util.spec_from_file_location('v39_component',module_path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
component=module.ContrastiveReadout(find_base(dict(model_type='qwen3',hidden_size=4096,num_hidden_layers=36)),str(v39_root/'adapter'),layers=v39_manifest['layers'],max_length=v39_manifest['max_length'],batch_size=v39_manifest['batch_size'])
features=component.features(test.text.tolist())
labels,scores=component.predict(features,v39_root/'readout.npz')
assert set(labels)==set(LABELS) and len(set(labels))==K
scores=scores[:,[list(labels).index(label) for label in LABELS]]
assert scores.shape==LOGIT.shape and np.isfinite(scores).all()
V39_BASE_LOGIT=LOGIT.copy()
vote('ce8b_v39',softmax(scores),0.15)
np.savez_compressed('v39_scores.npz',ids=test.id.to_numpy(dtype=str),labels=np.asarray(LABELS),baseline_scores=V39_BASE_LOGIT,component_scores=scores,combined_scores=LOGIT)
del component,features;gc.collect();torch.cuda.empty_cache()
'''
    final_audit=f'''assert set(m['member'] for m in MEMBER_AUDIT)>={{'svc','lora_qwen3_14b','lora_qwen25_14b','deberta','ce8b_v39'}}
assert not any(m['member']=='contrastive_v21' for m in MEMBER_AUDIT)
assert np.isfinite(LOGIT).all() and np.isfinite(P).all() and np.allclose(P.sum(1),1,atol=1e-5)
assert sub.columns.tolist()==['id','model'] and sub.id.tolist()==test.id.astype(str).tolist()
assert sub.id.is_unique and sub.model.isin(LABELS).all() and pd.read_parquet('submission.parquet').equals(sub)
audit={{'experiment':'v39','rows':len(test),'members':MEMBER_AUDIT,'elapsed_seconds':time.time()-V39_STARTED,'manifest_sha256':V39_MANIFEST_SHA256,'runtime_packages':V39_VERSIONS,'component_weight':0.15,'neural_training_at_inference':False,'claimed_public_gain':None,'claimed_full_ensemble_cv':None,'attribution':v39_manifest['attribution']}}
Path('v39-audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
signal.alarm(0)
'''
    setup=nbformat.read('notebooks/training-v34/training.ipynb',4).cells[1].source
    setup += """
import importlib.metadata as v39_metadata
V39_VERSIONS={p:v39_metadata.version(p) for p in ['torch','transformers','peft','bitsandbytes','numpy','scipy','scikit-learn']}
for package,version in {'transformers':'5.16.1','peft':'0.20.0','bitsandbytes':'0.50.0'}.items():
    assert V39_VERSIONS[package]==version,(package,V39_VERSIONS[package],version)
assert V39_VERSIONS['torch'].split('+')[0]=='2.11.0',V39_VERSIONS['torch']
print({'v39_runtime_packages':V39_VERSIONS},flush=True)
"""
    timer='''import time,signal
V39_STARTED=time.time()
def v39_timeout(signum,frame):raise TimeoutError('V39 inference runtime budget exhausted')
signal.signal(signal.SIGALRM,v39_timeout)
signal.alarm(31800)
'''
    nb.cells.insert(0,nbformat.v4.new_markdown_cell('# V39: fixed CE8B inference candidate\nAudited full-host weights, no neural fitting or conditional model selection at inference. Credited V16 baseline remains unchanged; a single fixed0.15 CE8B vote replaces the V21 4B addition. Original attributions follow. Full ensemble public improvement remains unproven.'))
    nb.cells.insert(1,nbformat.v4.new_code_cell(timer));nb.cells.insert(2,nbformat.v4.new_code_cell(setup))
    index=next(i for i,c in enumerate(nb.cells) if c.cell_type=='code' and c.source.startswith('def sinkhorn('))
    nb.cells.insert(index,nbformat.v4.new_code_cell(addition));nb.cells[-1]=nbformat.v4.new_code_cell(final_audit)
    assert [c.source for c in nb.cells if c.cell_type=='code' and c.source in codes]==[c for c in codes if c!=old_audit]
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,'v39','exec');c.outputs=[];c.execution_count=None
    nbformat.validate(nb);folder.mkdir(parents=True);nbformat.write(nb,folder/'inference.ipynb')
    meta=json.loads(Path('notebooks/improvement-v16/kernel-metadata.json').read_text())
    meta.update(id='adithyagiri/name-that-llm-ce8b-inference-v39',title='Name That LLM CE8B Inference V39',code_file='inference.ipynb')
    meta['dataset_sources'] += [dataset_id,'adithyagiri/name-that-llm-offline-quantization-deps']
    meta['model_sources'] += ['qwen-lm/qwen-3/Transformers/8b-base/1']
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet'] and not meta['kernel_sources']
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    receipt={'status':'built_not_executed','notebook_sha256':sha(folder/'inference.ipynb'),'manifest_sha256':sha(manifest_path),
             'dataset_bytes':sum(p.stat().st_size for p in dataset.rglob('*') if p.is_file()),'dataset_id':dataset_id,
             'requires_private_dataset_upload':True,'baseline_code_cells_preserved':len(codes)-1,
             'maximum_runtime_seconds':31800,'inference_helper_only':True,'public_score_claim':None,
             'remaining':['Private dataset upload with checksum readback','Successful GPU visible-row execution','ID/schema/finite-score audit','Actual hidden submission score']}
    (out/'build-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))

if __name__=='__main__':build()
