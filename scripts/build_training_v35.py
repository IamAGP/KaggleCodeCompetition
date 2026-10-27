"""Freeze a matched host-sampling comparison; no execution or submission."""
import copy
import hashlib
import json
from pathlib import Path
import nbformat
from build_training_v34 import sha, training_source, readout_source, replace_once


def build():
    root = Path('artifacts/improvement-v35')
    folder = Path('notebooks/training-v35')
    assert not folder.exists() and not root.exists(), 'Never overwrite frozen V35'
    protocol = json.loads(Path('configs/name-that-llm/v35.json').read_text())
    base = json.loads(Path('artifacts/improvement-v34/frozen-plan.json').read_text())
    assert sha(protocol['primary_folds']['file']) == protocol['primary_folds']['sha256']
    spec = copy.deepcopy(base['arms']['4b_small_supcon'])
    assert sha('scripts/train_adapter_v18.py') == spec['origin_worker_sha256']
    assert sha('scripts/readout_v19.py') == spec['origin_readout_sha256']
    checkpoint = Path('artifacts/improvement-v18/kaggle-output/ce_supcon/external_only')
    for relative, digest in spec['checkpoint_sha256'].items():
        assert sha(checkpoint/relative) == digest
    plan = {k: copy.deepcopy(base[k]) for k in ['host_sha256','labels','fold_by_prompt','fold_sizes']}
    plan.update(experiment='improvement-v35', protocol=protocol,
                protocol_sha256=sha('configs/name-that-llm/v35.json'), arms={})
    sources = {name: Path('scripts',name).read_text() for name in
               ['run_confirmation_arm_v35.py','prepare_likelihood_v33.py','sampling_v35.py']}
    worker = training_source('scripts/train_adapter_v18.py')
    for arm in ['independent', 'crossed']:
        item = copy.deepcopy(spec)
        item['parameters']['wall_limit_seconds'] = protocol['runtime']['wall_limit_seconds']
        item['training_worker'] = arm+'_worker.py'
        item['readout_worker'] = arm+'_readout.py'
        source = worker
        if arm == 'crossed':
            source = replace_once(source, '\nclass AttributionModel',
                                  '\nfrom sampling_v35 import crossed_batches as paired_batches\n\nclass AttributionModel')
        sources[item['training_worker']] = source
        sources[item['readout_worker']] = readout_source('scripts/readout_v19.py')
        plan['arms'][arm] = item
    setup = nbformat.read('notebooks/training-v34/training.ipynb',4).cells[1].source
    plan['source_sha256'] = {name: hashlib.sha256(s.encode()).hexdigest() for name,s in sources.items()}
    plan['setup_sha256'] = hashlib.sha256(setup.encode()).hexdigest()
    code = "import os, sys, subprocess, json, hashlib, time, signal\nfrom pathlib import Path\n"
    code += "for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'\n"
    code += "os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')\n"
    for name, source in sources.items():
        compile(source,name,'exec')
        code += f'Path({name!r}).write_text({source!r})\n'
    code += f'plan=json.loads({json.dumps(json.dumps(plan))})\n'
    code += '''
import torch
assert torch.cuda.device_count()==2
hosts=list(Path('/kaggle/input').rglob('train.csv'))
assert len(hosts)==1 and hashlib.sha256(hosts[0].read_bytes()).hexdigest()==plan['host_sha256']
plan['host_path']=str(hosts[0])
for arm,spec in plan['arms'].items():
    models=[]
    for p in Path('/kaggle/input').rglob('config.json'):
        cfg=json.loads(p.read_text())
        if cfg.get('model_type')=='qwen3' and cfg.get('hidden_size')==2560 and cfg.get('num_hidden_layers')==36:
            models.append(p.parent)
    assert len(models)==1,models
    spec['model_dir']=str(models[0])
    weights=[p for p in Path('/kaggle/input').rglob('adapter_model.safetensors')
             if p.parent.parent.name=='external_only' and p.parent.parent.parent.name=='ce_supcon'
             and hashlib.sha256(p.read_bytes()).hexdigest()==spec['checkpoint_sha256']['adapter/adapter_model.safetensors']]
    assert len(weights)==1
    spec['checkpoint']=str(weights[0].parent.parent)
    for relative,digest in spec['checkpoint_sha256'].items():
        assert hashlib.sha256((Path(spec['checkpoint'])/relative).read_bytes()).hexdigest()==digest
plan['deadline_unix']=time.time()+plan['protocol']['runtime']['wall_limit_seconds']
Path('plan.json').write_text(json.dumps(plan,indent=2))
processes=[]
try:
    for gpu,arm in enumerate(plan['arms']):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
        processes.append(subprocess.Popen([sys.executable,'-u','run_confirmation_arm_v35.py',arm],env=env,start_new_session=True))
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('V35 arm failed')
        if time.time()>plan['deadline_unix']:raise TimeoutError('V35 hard deadline')
        time.sleep(10)
    assert all(p.returncode==0 for p in processes)
finally:
    for p in processes:
        try: os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError: pass
    time.sleep(2)
    for p in processes:
        try: os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError: pass
    for p in processes:p.wait()
Path('summary.json').write_text(json.dumps({'completed':True,'experiment':'improvement-v35','submission_authorized':False}))
'''
    compile(code,'v35-notebook','exec')
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        '# V35: controlled host prompt sampling\nTwo matched Qwen3-4B arms, two fixed folds. '
        'SupCon paper https://arxiv.org/abs/2004.11362 and author implementation '
        'https://github.com/HobbitLong/SupContrast. Crossed prompt sampling is our untested extension. '
        'No automatic submission; see frozen protocol.'),nbformat.v4.new_code_cell(setup),nbformat.v4.new_code_cell(code)],
        metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbformat.validate(notebook)
    metadata=json.loads(Path('notebooks/training-v34/kernel-metadata.json').read_text())
    metadata.update(id='adithyagiri/name-that-llm-prompt-control-v35', title='Name That LLM Prompt Control V35',
                    kernel_sources=['adithyagiri/name-that-llm-contrastive-v18'],
                    model_sources=['qwen-lm/qwen-3/Transformers/4b-base/1'])
    folder.mkdir(parents=True); root.mkdir(parents=True)
    nbformat.write(notebook,folder/'training.ipynb')
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    plan['notebook_sha256']=sha(folder/'training.ipynb')
    (root/'frozen-plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({'notebook':str(folder),'sha256':plan['notebook_sha256'],'launched':False}))

if __name__=='__main__':build()
