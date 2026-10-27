"""Private offline conditional confirmation, full-host fitting and scored inference."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import nbformat


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build():
    out=Path('artifacts/improvement-v37');out.mkdir(exist_ok=False)
    inputs=set(json.loads(Path('artifacts/improvement-v34/evaluation-inputs.json').read_text()))
    inputs.update(['artifacts/improvement-v34/evaluation-inputs.json','artifacts/improvement-v34/primary-audit.json',
        'artifacts/improvement-v34/primary-oof.npz','artifacts/improvement-v34/related/frozen-plan.json',
        'artifacts/improvement-v4/robustness/folds.csv','configs/name-that-llm/v32.json',
        'data/name-that-llm/external/pool_v4.parquet','artifacts/improvement-v17/selected-pool-ids.csv'])
    script_names=['evaluate_related_v34','evaluate_v34','prepare_likelihood_v33','diagnose_balance_transfer',
        'build_related_v34','build_training_v34','build_full_v34','audit_full_v34',
        'train_adapter_v25','train_full_v26','component_v26']
    inputs.update('scripts/'+name+'.py' for name in script_names)
    manifest={'files':[{'path':p,'bytes':Path(p).stat().st_size,'sha256':sha(p)} for p in sorted(inputs)],'private':True}
    payload=out/'v37-evaluation-payload.tar.gz'
    if os.environ.get('V37_EXISTING_PAYLOAD'):
        shutil.copyfile(os.environ['V37_EXISTING_PAYLOAD'],payload)
        with tarfile.open(payload) as archive:
            assert sorted(m.name for m in archive.getmembers() if m.isfile())==sorted(inputs)
            for entry in manifest['files']:
                data=archive.extractfile(entry['path']).read()
                assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
    else:
        with tarfile.open(payload,'w:gz') as t:
            for p in sorted(inputs):t.add(p,arcname=p)
    (out/'payload-manifest.json').write_text(json.dumps(manifest,indent=2))
    dataset={'title':'Name That LLM V37 Private Frozen Evaluation Inputs','id':'adithyagiri/name-that-llm-v37-private-evaluation-inputs',
             'licenses':[{'name':'CC-BY-SA-4.0'}]}
    (out/'dataset-metadata.json').write_text(json.dumps(dataset,indent=2))
    related=nbformat.read('notebooks/training-v34-related/training.ipynb',4)
    baseline=nbformat.read('notebooks/improvement-v16/name-that-llm-two-stage-v16.ipynb',4)
    v21=nbformat.read('notebooks/improvement-v21/inference.ipynb',4)
    fallback=next(c.source for c in v21.cells if c.cell_type=='code' and 'V21_ASSET_HASHES=' in c.source)
    config=json.loads(Path('configs/name-that-llm/v16.json').read_text())
    assert sha('notebooks/improvement-v16/name-that-llm-two-stage-v16.ipynb')==config['notebook_sha256']
    lifecycle=f'''import os,sys,json,hashlib,subprocess,time,shutil
from pathlib import Path
V37_ROOT=Path.cwd()
def remaining_budget():
    remaining=V37_DEADLINE-time.time()
    assert remaining>0,'V37 global runtime exhausted'
    return remaining
def run_stage(command,check=True,timeout=None,**kwargs):
    import psutil
    process=subprocess.Popen(command,**kwargs)
    tracked={{}};started=time.monotonic()
    try:
        while True:
            try:
                for child in psutil.Process(process.pid).children(recursive=True):tracked[child.pid]=child
            except psutil.NoSuchProcess:pass
            remaining=None if timeout is None else timeout-(time.monotonic()-started)
            if remaining is not None and remaining<=0:raise subprocess.TimeoutExpired(command,timeout)
            try:
                status=process.wait(timeout=.5 if remaining is None else min(.5,remaining));break
            except subprocess.TimeoutExpired:continue
        if check and status:raise subprocess.CalledProcessError(status,command)
        return status
    except BaseException:
        # Capture descendants before killing the runner, including separate sessions.
        try:
            for child in psutil.Process(process.pid).children(recursive=True):tracked[child.pid]=child
        except psutil.NoSuchProcess:pass
        descendants=list(tracked.values())
        for child in reversed(descendants):
            try:child.terminate()
            except psutil.NoSuchProcess:pass
        if process.poll() is None:process.terminate()
        _,alive=psutil.wait_procs(descendants,timeout=3)
        for child in alive:
            try:child.kill()
            except psutil.NoSuchProcess:pass
        if process.poll() is None:process.kill()
        process.wait()
        psutil.wait_procs(alive,timeout=3)
        raise
manifest={manifest!r}
marker='artifacts/improvement-v34/evaluation-inputs.json'
marker_hash=next(e['sha256'] for e in manifest['files'] if e['path']==marker)
auxiliary=[p.parents[2] for p in Path('/kaggle/input').rglob('evaluation-inputs.json')
           if hashlib.sha256(p.read_bytes()).hexdigest()==marker_hash and p.as_posix().endswith(marker)]
assert len(auxiliary)==1,auxiliary
V37_AUXILIARY=auxiliary[0]
for entry in manifest['files']:
    source=V37_AUXILIARY/entry['path'];assert source.stat().st_size==entry['bytes']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==entry['sha256'],str(source)
    destination=V37_ROOT/entry['path'];assert not destination.exists()
    destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
def filtered_inputs(source,destination):
    destination.mkdir()
    for child in source.iterdir():
        if child==V37_AUXILIARY:continue
        target=destination/child.name
        # pathlib.rglob does not recurse directory symlinks: mirror directories,
        # and symlink only individual files without copying large model weights.
        if child.is_dir():filtered_inputs(child,target)
        else:target.symlink_to(child)
V37_TRAINING_INPUT=V37_ROOT/'v37-training-inputs'
filtered_inputs(Path('/kaggle/input'),V37_TRAINING_INPUT)
assert len(list(V37_TRAINING_INPUT.rglob('train.csv')))==1
def linked_output(experiment,destination):
    candidates=[]
    for p in Path('/kaggle/input').rglob('plan.json'):
        try: v=json.loads(p.read_text())
        except (ValueError,OSError):continue
        if v.get('experiment')==experiment and (experiment!='improvement-v34' or p.parent.joinpath('summary.json').exists()):
            candidates.append(p.parent)
    assert len(candidates)==1,(experiment,candidates)
    dst=Path(destination);dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():dst.symlink_to(candidates[0],target_is_directory=True)
    else:
        # Pilot NPZ is already staged here; overlay attached immutable outputs.
        for source in candidates[0].iterdir():
            target=dst/source.name
            if target.exists():
                assert target.is_file() and source.is_file()
                assert hashlib.sha256(target.read_bytes()).hexdigest()==hashlib.sha256(source.read_bytes()).hexdigest()
            else:target.symlink_to(source,target_is_directory=source.is_dir())
linked_output('improvement-v18','artifacts/improvement-v18/kaggle-output')
linked_output('improvement-v25','artifacts/improvement-v25/kaggle-output')
linked_output('improvement-v34','artifacts/improvement-v34/kaggle-output')
related_code={related.cells[2].source!r}
assert related_code.count("Path('/kaggle/input')")==3
related_code=related_code.replace("Path('/kaggle/input')",f"Path({{str(V37_TRAINING_INPUT)!r}})")
related_output=V37_ROOT/'artifacts/improvement-v34/related/kaggle-output'
related_output.mkdir(exist_ok=False)
Path('v37-related-run.py').write_text(related_code)
run_stage([sys.executable,'-u',str(V37_ROOT/'v37-related-run.py')],cwd=related_output,check=True,timeout=min(9000,remaining_budget()))
run_stage([sys.executable,'-u','scripts/evaluate_related_v34.py'],check=True,timeout=min(1800,remaining_budget()))
V37_GATE=json.loads(Path('artifacts/improvement-v34/related/evaluation-audit.json').read_text())
assert V37_GATE['verified']
V37_GATE_PASSED=bool(V37_GATE['qualifies_for_full_host_training'])
if V37_GATE_PASSED:
    run_stage([sys.executable,'-u','scripts/build_full_v34.py'],check=True,timeout=min(120,remaining_budget()))
    V37_FULL_ROOT=V37_ROOT/'artifacts/improvement-v34/full-host/kaggle-output'
    V37_FULL_ROOT.mkdir(exist_ok=False)
    env=dict(os.environ,V34_INPUT_ROOT=str(V37_TRAINING_INPUT))
    run_stage([sys.executable,'-u',str(V37_ROOT/'artifacts/improvement-v34/full-host/run.py')],cwd=V37_FULL_ROOT,env=env,check=True,timeout=min(9300,remaining_budget()))
    run_stage([sys.executable,'-u','scripts/audit_full_v34.py'],check=True,timeout=min(300,remaining_budget()))
    assert json.loads(Path('artifacts/improvement-v34/full-host/training-audit.json').read_text())['verified']
Path('v37-route.json').write_text(json.dumps({{'related_gate_passed':V37_GATE_PASSED,'route':'new8b' if V37_GATE_PASSED else 'incumbent_v21','elapsed_seconds':time.time()-V37_STARTED,'hidden_labels_used':False}},indent=2))
'''
    addition=f'''if not V37_GATE_PASSED:
    exec({fallback!r},globals())
    V37_SELECTED_MEMBER='contrastive_v21'
else:
    import importlib.util
    module_path=V37_FULL_ROOT/'component_v34.py'
    spec=importlib.util.spec_from_file_location('v37_component',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    plan=json.loads((V37_FULL_ROOT/'plan.json').read_text())
    assert hashlib.sha256(module_path.read_bytes()).hexdigest()==plan['source_sha256']['component_v34.py']
    component=module.ContrastiveReadout(find_base(dict(model_type='qwen3',hidden_size=4096,num_hidden_layers=36)),str(V37_FULL_ROOT/'full/adapter'),layers=[12,24,36],max_length=plan['training_parameters']['max_length'],batch_size=8)
    features=component.features(test.text.tolist())
    labels,scores=component.predict(features,V37_FULL_ROOT/'full/readout.npz')
    assert set(labels)==set(LABELS) and len(labels)==K
    scores=scores[:,[list(labels).index(label) for label in LABELS]]
    assert scores.shape==LOGIT.shape and np.isfinite(scores).all()
    vote('ce8b_v37',softmax(scores),0.15)
    np.savez_compressed('v37_component_scores.npz',ids=test.id.to_numpy(dtype=str),labels=np.asarray(LABELS),scores=scores)
    del component,features;gc.collect();torch.cuda.empty_cache()
    V37_SELECTED_MEMBER='ce8b_v37'
'''
    audit='''assert set(m['member'] for m in MEMBER_AUDIT)>={'svc','lora_qwen3_14b','lora_qwen25_14b','deberta',V37_SELECTED_MEMBER}
assert np.isfinite(LOGIT).all() and np.isfinite(P).all() and np.allclose(P.sum(1),1,atol=1e-5)
assert sub.columns.tolist()==['id','model'] and sub.id.tolist()==test.id.astype(str).tolist()
assert sub.id.is_unique and sub.model.isin(LABELS).all() and pd.read_parquet('submission.parquet').equals(sub)
audit={'experiment':'v37','route':'new8b' if V37_GATE_PASSED else 'incumbent_v21','related_gate':V37_GATE,'rows':len(test),'members':MEMBER_AUDIT,'elapsed_seconds':time.time()-V37_STARTED,'claimed_public_gain':None,'hidden_labels_used':False,'selection':'fixed training-only related gate; no hidden score selection','attribution':'Credited V16 ensemble: Nishant Dahal and contributors. Qwen backbones; established LoRA/SupCon/QLoRA methods.'}
Path('v37-audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
'''
    baseline.cells.insert(0,nbformat.v4.new_markdown_cell('# V37: frozen conditional delivery\nPrivate offline training-only related gate; if passed, full-host 8B CE; otherwise fixed V21 fallback. No hidden-label selection and no claimed score gain. Original V16 attribution retained.'))
    baseline.cells.insert(1,nbformat.v4.new_code_cell(related.cells[1].source))
    baseline.cells.insert(2,nbformat.v4.new_code_cell(lifecycle))
    baseline.cells.insert(1,nbformat.v4.new_code_cell('''import time,signal
V37_STARTED=time.time()
V37_DEADLINE=V37_STARTED+31800
def v37_timeout(signum,frame):raise TimeoutError('V37 global runtime budget exhausted')
signal.signal(signal.SIGALRM,v37_timeout)
signal.alarm(31800)
'''))
    index=next(i for i,c in enumerate(baseline.cells) if c.cell_type=='code' and c.source.startswith('def sinkhorn('))
    baseline.cells.insert(index,nbformat.v4.new_code_cell(addition));baseline.cells[-1]=nbformat.v4.new_code_cell(audit)
    for c in baseline.cells:
        if c.cell_type=='code':compile(c.source,'v37','exec');c.outputs=[];c.execution_count=None
    nbformat.validate(baseline)
    folder=Path('notebooks/improvement-v37');folder.mkdir(exist_ok=False)
    nbformat.write(baseline,folder/'inference.ipynb')
    metadata=json.loads(Path('notebooks/improvement-v21/kernel-metadata.json').read_text())
    metadata.update(id='adithyagiri/name-that-llm-conditional-delivery-v37',title='Name That LLM Conditional Delivery V37')
    metadata['dataset_sources'] += ['adithyagiri/name-that-llm-offline-quantization-deps',dataset['id']]
    metadata['model_sources'] += ['qwen-lm/qwen-3/Transformers/8b-base/1']
    metadata['kernel_sources'] += ['adithyagiri/name-that-llm-contrastive-v18','adithyagiri/name-that-llm-qlora-training-v25','adithyagiri/name-that-llm-fold-confirmation-v34']
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    receipt={'payload_bytes':payload.stat().st_size,'payload_sha256':sha(payload),'notebook_sha256':sha(folder/'inference.ipynb'),
        'status':'built_not_gpu_executed','requires_private_dataset_upload':dataset['id'],'protocol_unchanged':True,
        'runtime_estimate_before_base_inference_seconds':11500,'maximum_runtime_seconds':32400,'launch_or_submission_performed':False}
    (out/'build-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':build()
