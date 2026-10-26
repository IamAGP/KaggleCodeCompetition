"""Package fixed host-fold comparisons using audited V18/V25 external weights."""
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from safetensors.numpy import load_file


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, old, new):
    assert source.count(old)==1, old
    return source.replace(old,new,1)


def training_source(path):
    source = Path(path).read_text()
    needle = '    trainable = [p for p in model.parameters() if p.requires_grad]'
    resume = '''    # Restore only an independently verified external-only checkpoint.
    import hashlib
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    checkpoint = Path(plan['checkpoint'])
    for relative,digest in plan['checkpoint_sha256'].items():
        assert hashlib.sha256((checkpoint/relative).read_bytes()).hexdigest()==digest
    state = load_file(str(checkpoint/'adapter/adapter_model.safetensors'))
    restored = set_peft_model_state_dict(model.encoder,state)
    assert not restored.unexpected_keys, restored.unexpected_keys
    assert not [k for k in restored.missing_keys if 'lora_' in k], restored.missing_keys
    heads = torch.load(checkpoint/'heads.pt',map_location='cpu',weights_only=True)
    for name in ['norm','head','projection']:
        getattr(model,name).load_state_dict(heads[name],strict=True)
    assert set(frame.stage)=={'host'}
    assert len(frame)==plan['training_rows'] and len(valid)==plan['validation_rows']
    assert frame.id.is_unique and valid.id.is_unique and set(frame.id).isdisjoint(valid.id)
    log(event='external_checkpoint_loaded',sha256=plan['checkpoint_sha256'],
        fit_rows=len(frame),validation_rows=len(valid))
'''
    source = replace_once(source,needle,resume+needle)
    source = replace_once(source,
        'for stage, epochs in [("pool", plan["pool_epochs"]), ("host", plan["host_epochs"])]:',
        'for stage, epochs in [("host", plan["host_epochs"])]:')
    compile(source,'v34-host-worker','exec')
    return source


def readout_source(path):
    source = Path(path).read_text()
    source = replace_once(source,'assert len(fit_ids) == 2159 and len(val_ids) == 540',
        'assert len(fit_ids) == plan["training_rows"] and len(val_ids) == plan["validation_rows"]')
    compile(source,'v34-readout-worker','exec')
    return source


def build():
    protocol_path = Path('configs/name-that-llm/v34.json')
    protocol = json.loads(protocol_path.read_text())
    fold_path = Path(protocol['primary_folds']['file'])
    assert sha(fold_path)==protocol['primary_folds']['sha256']
    host_path = Path('data/name-that-llm/train.csv')
    host = pd.read_csv(host_path,dtype={'id':str,'prompt_id':str})
    folds = pd.read_csv(fold_path,dtype={'id':str,'prompt_id':str})
    assert folds.id.equals(host.id) and folds.prompt_id.equals(host.prompt_id)
    assert folds.groupby('prompt_id').fold.nunique().max()==1
    mapping = {p:{'group':'host:'+p,'fold':int(f)} for p,f in folds.groupby('prompt_id').fold.first().items()}
    plan = {'experiment':'improvement-v34','protocol':protocol,'protocol_sha256':sha(protocol_path),
            'host_sha256':sha(host_path),'labels':sorted(host.model.unique()),'fold_by_prompt':mapping,
            'fold_sizes':{str(f):{'training_rows':int((folds.fold!=f).sum()),
                                  'validation_rows':int((folds.fold==f).sum())} for f in [1,2,3,4]},'arms':{}}
    sources = {name:Path('scripts',name).read_text() for name in ['run_confirmation_arm_v34.py','prepare_likelihood_v33.py']}
    definitions = [
        ('8b_ce',25,'ce','ce_final','scripts/train_adapter_v25.py','scripts/readout_v25.py',4096,504),
        ('4b_small_supcon',18,'ce_supcon','supcon_final','scripts/train_adapter_v18.py','scripts/readout_v19.py',2560,144),
    ]
    for arm,version,variant,readout_variant,worker,readout,width,count in definitions:
        origin = Path(f'artifacts/improvement-v{version}/kaggle-output')
        frozen = json.loads(Path(f'artifacts/improvement-v{version}/frozen-plan.json').read_text())
        params = json.loads((origin/'plan.json').read_text())
        assert sha(origin/'worker.py')==frozen['worker_sha256']==sha(worker)
        for key in ['seed','host_epochs','host_adapter_lr','host_head_lr','batch_size','gradient_accumulation',
                    'lora_rank','lora_alpha','target_modules','max_length','contrastive_weight','contrastive_temperature']:
            assert params[key]==frozen[key], key
        assert params['host_sha256']==plan['host_sha256'] and params['labels']==plan['labels']
        frame = pd.read_parquet(origin/'training_frame.parquet')
        assert (frame.stage=='pool').sum()==(18000 if version==25 else 4500)
        assert set(frame.loc[frame.stage=='pool','group']).isdisjoint(frame.loc[frame.stage=='host','group'])
        log = [json.loads(line) for line in (origin/variant/'progress.jsonl').read_text().splitlines()]
        epochs = [r for r in log if r['event']=='epoch_complete']
        assert [(r['stage'],r['epoch']) for r in epochs]==[('pool',0),('host',0),('host',1)]
        assert log[-1]['event']=='complete'
        checkpoint = origin/variant/'external_only'
        hashes = {str(p.relative_to(checkpoint)):sha(p) for p in checkpoint.rglob('*') if p.is_file()}
        tensors = load_file(str(checkpoint/'adapter/adapter_model.safetensors'))
        assert len(tensors)==count and all(np.isfinite(v).all() for v in tensors.values())
        assert sha(checkpoint/'adapter/adapter_model.safetensors')!=sha(origin/variant/'final/adapter/adapter_model.safetensors')
        config = json.loads((checkpoint/'adapter/adapter_config.json').read_text())
        assert config['r']==16 and config['lora_alpha']==32
        import torch
        heads = torch.load(checkpoint/'heads.pt',map_location='cpu',weights_only=True)
        assert heads['head']['weight'].shape==(18,width)
        assert all(torch.isfinite(t).all() for state in heads.values() for t in state.values())
        training_name = arm+'_worker.py'; readout_name = arm+'_readout.py'
        sources[training_name] = training_source(worker)
        sources[readout_name] = readout_source(readout)
        # Drop runtime locations from the originating notebook; re-resolve by verified hashes.
        params.pop('model_dir',None)
        params['wall_limit_seconds']=protocol['runtime']['wall_limit_seconds']
        settings = {'layers':[12,24,36],'C':.02,'max_iter':1000,'seed':1901,'batch_size':8}
        plan['arms'][arm] = {'parameters':params,'variant':variant,'readout_variant':readout_variant,
            'training_worker':training_name,'readout_worker':readout_name,'readout':settings,
            'checkpoint_sha256':hashes,'hidden_size':width,'origin_version':version,
            'origin_worker_sha256':sha(worker),'origin_readout_sha256':sha(readout)}
    plan['source_sha256'] = {name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}
    code = "import os, sys, subprocess, json, hashlib, time, signal\nfrom pathlib import Path\n"
    code += "for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='2'\n"
    code += "os.environ['HF_HUB_OFFLINE']='1'\nos.environ['TRANSFORMERS_OFFLINE']='1'\nos.environ['TOKENIZERS_PARALLELISM']='false'\n"
    for name,source in sources.items():code+=f'Path({name!r}).write_text({source!r})\n'
    code += f'plan=json.loads({json.dumps(json.dumps(plan))})\n'
    code += '''
import torch
assert torch.cuda.device_count()==2
hosts=list(Path('/kaggle/input').rglob('train.csv'))
assert len(hosts)==1
assert hashlib.sha256(hosts[0].read_bytes()).hexdigest()==plan['host_sha256']
plan['host_path']=str(hosts[0])
for arm,spec in plan['arms'].items():
    candidates=[]
    for p in Path('/kaggle/input').rglob('config.json'):
        cfg=json.loads(p.read_text())
        if cfg.get('model_type')=='qwen3' and cfg.get('hidden_size')==spec['hidden_size'] and cfg.get('num_hidden_layers')==36:
            candidates.append(p.parent)
    assert len(candidates)==1,(arm,candidates)
    spec['model_dir']=str(candidates[0])
    weights=[p for p in Path('/kaggle/input').rglob('adapter_model.safetensors')
             if p.parent.parent.name=='external_only' and p.parent.parent.parent.name==spec['variant']
             and hashlib.sha256(p.read_bytes()).hexdigest()==spec['checkpoint_sha256']['adapter/adapter_model.safetensors']]
    assert len(weights)==1,(arm,len(weights))
    spec['checkpoint']=str(weights[0].parent.parent)
    for relative,digest in spec['checkpoint_sha256'].items():
        assert hashlib.sha256((Path(spec['checkpoint'])/relative).read_bytes()).hexdigest()==digest
plan['deadline_unix']=time.time()+plan['protocol']['runtime']['wall_limit_seconds']
Path('plan.json').write_text(json.dumps(plan,indent=2))
processes=[]
try:
    for gpu,arm in enumerate(plan['arms']):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
        processes.append(subprocess.Popen([sys.executable,'-u','run_confirmation_arm_v34.py',arm],env=env,start_new_session=True))
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Confirmation arm failed')
        if time.time()>plan['deadline_unix']:raise TimeoutError('Confirmation notebook wall budget exhausted')
        time.sleep(10)
    assert all(p.returncode==0 for p in processes)
except BaseException:
    for p in processes:
        if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
    for p in processes:p.wait()
    raise
summary={'completed':True,'experiment':'improvement-v34','submission_authorized':False,
         'decision':'Independent multi-fold audit and frozen gates required'}
Path('summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
'''
    compile(code,'v34-notebook','exec')
    setup = nbformat.read('notebooks/training-v25/training.ipynb',4).cells[1].source
    plan['setup_sha256']=hashlib.sha256(setup.encode()).hexdigest()
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        '# V34: broader confirmation with fixed class balancing\n'
        '8B CE versus the 4B small-pool SupCon recipe used by V21. Four independent host folds per arm, '
        'each initialized from verified external-only weights. No submission. '
        'Public pool and balancing credited to Nishant Dahal and contributors. '
        'LoRA: Hu et al. 2106.09685; SupCon: Khosla et al. 2004.11362; '
        'QLoRA: Dettmers et al. 2305.14314. Original V25 and V30 failed gates remain unchanged.'),
        nbformat.v4.new_code_cell(setup),nbformat.v4.new_code_cell(code)],
        metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    folder=Path('notebooks/training-v34');assert not folder.exists(),'Do not overwrite a frozen package'
    folder.mkdir(parents=True);nbformat.validate(nb);nbformat.write(nb,folder/'training.ipynb')
    meta={'id':'adithyagiri/name-that-llm-fold-confirmation-v34','title':'Name That LLM Fold Confirmation V34',
          'code_file':'training.ipynb','language':'python','kernel_type':'notebook','is_private':True,
          'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4',
          'competition_sources':['name-that-llm'],
          'dataset_sources':['adithyagiri/name-that-llm-offline-quantization-deps'],
          'kernel_sources':['adithyagiri/name-that-llm-contrastive-v18','adithyagiri/name-that-llm-qlora-training-v25'],
          'model_sources':['qwen-lm/qwen-3/Transformers/4b-base/1','qwen-lm/qwen-3/Transformers/8b-base/1']}
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    plan['notebook_sha256']=sha(folder/'training.ipynb')
    root=Path('artifacts/improvement-v34');root.mkdir(exist_ok=True)
    (root/'frozen-plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({'folder':str(folder),'notebook_sha256':plan['notebook_sha256'],'arms':list(plan['arms'])}))


if __name__ == '__main__':
    build()
