"""Add our full-host contrastive member to the frozen, credited v16 ensemble."""
import hashlib
import json
from pathlib import Path
import nbformat


def build():
    plan=json.loads(Path('artifacts/improvement-v21/frozen-plan.json').read_text())
    output=Path('artifacts/improvement-v21/kaggle-output')
    trained=json.loads((output/'full/training_result.json').read_text())
    readout=json.loads((output/'full/readout_result.json').read_text())
    audit=json.loads(Path('artifacts/improvement-v21/training-audit.json').read_text())
    assert audit['verified'] and audit['training']==trained and audit['readout']==readout
    assert trained['completed'] and trained['rows']==2699 and trained['prompts']==150
    assert trained['labels']==plan['labels']==readout['labels'] and readout['fit_rows']==2699
    assert trained['epochs']==2 and trained['changed_adapter_tensors']>0
    assert readout['portable_readout_check']=='passed'
    source=Path('scripts/component_v21.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==plan['source_sha256']['component_v21.py']
    assets=['full/adapter/adapter_model.safetensors','full/adapter/adapter_config.json',
            'full/readout.npz','full/training_result.json','full/readout_result.json']
    hashes={p:hashlib.sha256((output/p).read_bytes()).hexdigest() for p in assets}
    for p in ['full/adapter/adapter_model.safetensors','full/readout.npz']:
        assert hashes[p]==readout['artifacts_sha256'][p]
    original=Path('notebooks/improvement-v16/name-that-llm-two-stage-v16.ipynb')
    baseline=json.loads(Path('configs/name-that-llm/v16.json').read_text())
    assert hashlib.sha256(original.read_bytes()).hexdigest()==baseline['notebook_sha256']
    nb=nbformat.read(original,4)
    original_codes=[c.source for c in nb.cells if c.cell_type=='code']
    original_audit=nb.cells[-1].source
    nb.cells.insert(0,nbformat.v4.new_markdown_cell(
        '# V21: prospective contrastive addition to the 0.620 ensemble\n\n'
        'The credited v16 ensemble below is retained with its original weights. '
        'Our full-host Qwen3-4B contrastive adapter adds a fixed 0.15 log-probability vote before '
        'the original balancing step. The component improved a matched local control; its complementarity '
        'with this ensemble has not been validated. This is a prospective submission, not a claimed CV gain. '
        'Training: [our V21 notebook](https://www.kaggle.com/code/adithyagiri/name-that-llm-full-contrastive-v21). '
        'LoRA and SupCon are established methods; our combination and readout experiments are credited separately '
        'from Nishant Dahal and the original public-ensemble and data contributors.'))
    setup="""
import subprocess,sys,importlib.metadata
from packaging.version import Version
try: torchao_version=importlib.metadata.version('torchao')
except importlib.metadata.PackageNotFoundError: torchao_version=None
if torchao_version is not None and Version(torchao_version)<Version('0.16.0'):
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True)
"""
    nb.cells.insert(1,nbformat.v4.new_code_cell(setup))
    addition=f"""
import hashlib
V21_ASSET_HASHES={hashes!r}
roots=[Path(p).parent for p in glob.glob('/kaggle/input/**/plan.json',recursive=True)
       if json.loads(Path(p).read_text()).get('experiment')=='improvement-v21']
assert len(roots)==1,roots
v21_root=roots[0]
for relative,digest in V21_ASSET_HASHES.items():
    assert hashlib.sha256((v21_root/relative).read_bytes()).hexdigest()==digest,relative
source={source!r}
assert hashlib.sha256(source.encode()).hexdigest()=={plan['source_sha256']['component_v21.py']!r}
Path('component_v21.py').write_text(source)
from component_v21 import ContrastiveReadout
v21_model_dir=find_base(dict(model_type='qwen3',hidden_size=2560,num_hidden_layers=36))
component=ContrastiveReadout(v21_model_dir,str(v21_root/'full/adapter'),layers={plan['readout']['layers']!r},
                            max_length={plan['training_parameters']['max_length']},batch_size={plan['readout']['batch_size']})
v21_features=component.features(test.text.tolist())
v21_labels,v21_scores=component.predict(v21_features,v21_root/'full/readout.npz')
assert set(v21_labels)==set(LABELS) and len(set(v21_labels))==K
v21_order=[list(v21_labels).index(label) for label in LABELS]
v21_scores=v21_scores[:,v21_order]
assert v21_scores.shape==LOGIT.shape and np.isfinite(v21_scores).all()
V21_INCUMBENT_LOGIT=LOGIT.copy()
vote('contrastive_v21',softmax(v21_scores),{plan['blend']['new_member_log_probability_weight']!r})
np.savez_compressed('v21_scores.npz',ids=test.id.to_numpy(dtype=str),labels=np.asarray(LABELS),
                    incumbent_scores=V21_INCUMBENT_LOGIT,component_scores=v21_scores,combined_scores=LOGIT)
del component,v21_features
gc.collect();torch.cuda.empty_cache()
"""
    index=next(i for i,c in enumerate(nb.cells) if c.cell_type=='code' and c.source.startswith('def sinkhorn('))
    nb.cells.insert(index,nbformat.v4.new_code_cell(addition))
    nb.cells[-1]=nbformat.v4.new_code_cell(f"""
assert set(m['member'] for m in MEMBER_AUDIT)>={{'svc','lora_qwen3_14b','lora_qwen25_14b','deberta','contrastive_v21'}}
assert np.isfinite(LOGIT).all() and np.isfinite(P).all()
assert np.allclose(P.sum(1),1,atol=1e-5)
assert sub.columns.tolist()==['id','model'] and sub.id.tolist()==test.id.astype(str).tolist()
assert sub.id.is_unique and sub.model.isin(LABELS).all()
assert pd.read_parquet('submission.parquet').equals(sub)
baseline_probs=softmax(V21_INCUMBENT_LOGIT)
if len(test)>=50: baseline_probs=sinkhorn(baseline_probs)
audit={{'experiment':'v21','base_public_score':0.620,'rows':len(test),'members':MEMBER_AUDIT,
        'changed_predictions_vs_v16':int((baseline_probs.argmax(1)!=P.argmax(1)).sum()),
        'component_weight':{plan['blend']['new_member_log_probability_weight']!r},'asset_sha256':V21_ASSET_HASHES,
        'elapsed_seconds':time.time()-T0,'claimed_ensemble_cv':None,
        'attribution':'Base ensemble: Nishant Dahal two-stage LoRA part two, original contributors retained; added component: our v18/v19/v21 experiments'}}
Path('v21_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit),flush=True)
""")
    preserved=[c.source for c in nb.cells if c.cell_type=='code' and c.source in original_codes]
    assert preserved==[s for s in original_codes if s!=original_audit]
    for cell in nb.cells:
        if cell.cell_type=='code':
            cell.outputs=[];cell.execution_count=None;compile(cell.source,'v21-inference','exec')
    nbformat.validate(nb)
    folder=Path('notebooks/improvement-v21');folder.mkdir(exist_ok=True)
    nbformat.write(nb,folder/'inference.ipynb')
    meta=json.loads(Path('notebooks/improvement-v16/kernel-metadata.json').read_text())
    meta.update(id='adithyagiri/name-that-llm-contrastive-blend-v21',title='Name That LLM Contrastive Blend V21',
                code_file='inference.ipynb',kernel_sources=['adithyagiri/name-that-llm-full-contrastive-v21'])
    meta['model_sources'].append(plan['model'])
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    plan.update(inference_notebook_sha256=hashlib.sha256((folder/'inference.ipynb').read_bytes()).hexdigest(),
                inference_asset_sha256=hashes,unchanged_baseline_code_cells=len(preserved))
    Path('artifacts/improvement-v21/inference-plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({'folder':str(folder),'unchanged_baseline_code_cells':len(preserved),
                      'notebook_sha256':plan['inference_notebook_sha256']}))


if __name__=='__main__':build()
