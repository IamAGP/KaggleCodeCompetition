"""Freeze and launchable-package full-host training for an additive submission."""
import hashlib
import json
from pathlib import Path
import nbformat


def build():
    plan=json.loads(Path('configs/name-that-llm/v21.json').read_text())
    upstream=Path('artifacts/improvement-v18/kaggle-output')
    original=json.loads((upstream/'plan.json').read_text())
    plan['training_parameters']={k:original[k] for k in ['seed','max_length','batch_size','gradient_accumulation',
        'lora_rank','lora_alpha','target_modules','host_epochs','host_adapter_lr','host_head_lr',
        'contrastive_weight','contrastive_temperature','wall_limit_seconds']}
    plan['labels']=original['labels'];plan['host_sha256']=original['host_sha256']
    plan['model']=original['model']
    plan['checkpoint_sha256']={p:hashlib.sha256((upstream/'ce_supcon/external_only'/p).read_bytes()).hexdigest()
                               for p in ['adapter/adapter_model.safetensors','adapter/adapter_config.json','heads.pt']}
    source_paths={'v18_worker.py':Path('scripts/train_adapter_v18.py'),
                  'train_full.py':Path('scripts/train_full_v21.py'),
                  'component_v21.py':Path('scripts/component_v21.py')}
    sources={name:p.read_text() for name,p in source_paths.items()}
    assert sources['v18_worker.py']==(upstream/'worker.py').read_text()
    plan['source_sha256']={name:hashlib.sha256(s.encode()).hexdigest() for name,s in sources.items()}
    code=f'''
import os,sys,json,glob,hashlib,subprocess,importlib.metadata
from pathlib import Path
from packaging.version import Version
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
try: version=importlib.metadata.version('torchao')
except importlib.metadata.PackageNotFoundError: version=None
if version is not None and Version(version)<Version('0.16.0'):
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=True)
import torch,peft,transformers
assert torch.cuda.device_count()>=1
print({{'torch':torch.__version__,'peft':peft.__version__,'transformers':transformers.__version__}},flush=True)
plan=json.loads({json.dumps(json.dumps(plan))})
matches=[Path(p).parent for p in glob.glob('/kaggle/input/**/plan.json',recursive=True)
         if json.loads(Path(p).read_text()).get('experiment')=='improvement-v18']
assert len(matches)==1,matches
plan['checkpoint']=str(matches[0]/'ce_supcon/external_only')
paths=[p for p in glob.glob('/kaggle/input/**/config.json',recursive=True) if '/4b-base/' in p.lower()]
assert len(paths)==1,paths
plan['model_dir']=str(Path(paths[0]).parent)
plan['host_path']=next(p for p in glob.glob('/kaggle/input/**/train.csv',recursive=True) if 'name-that-llm' in p)
assert hashlib.sha256(Path(plan['host_path']).read_bytes()).hexdigest()==plan['host_sha256']
Path('plan.json').write_text(json.dumps(plan,indent=2))
sources={sources!r}
for name,source in sources.items():
    assert hashlib.sha256(source.encode()).hexdigest()==plan['source_sha256'][name]
    Path(name).write_text(source)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
subprocess.run([sys.executable,'-u','train_full.py'],env=env,check=True)
subprocess.run([sys.executable,'-u','component_v21.py'],env=env,check=True)
assert json.loads(Path('full/training_result.json').read_text())['completed']
assert Path('full/readout.npz').is_file()
print({{'v21_training':'COMPLETE','submission_created':False}},flush=True)
'''
    compile(code,'v21','exec')
    for name,source in sources.items(): compile(source,name,'exec')
    folder=Path('notebooks/training-v21');folder.mkdir(exist_ok=True)
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        '# V21: full-host contrastive component for an additive submission\n\n'
        'Trains our v18 external-only checkpoint on all host prompts, then fits v19-style multilayer readouts. '
        'This notebook makes no validation claim and creates no submission. Separate inference adds the component '
        'to the credited Nishant Dahal v16 ensemble. LoRA: Hu et al.; SupCon: Khosla et al. '
        'Public pool credited to Nishant Dahal and original contributors, CC BY-SA 4.0.'),nbformat.v4.new_code_cell(code)])
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    nbformat.validate(nb);nbformat.write(nb,folder/'training.ipynb')
    meta={'id':'adithyagiri/name-that-llm-full-contrastive-v21','title':'Name That LLM Full Contrastive V21',
          'code_file':'training.ipynb','language':'python','kernel_type':'notebook','is_private':True,
          'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4',
          'competition_sources':['name-that-llm'],'dataset_sources':[],
          'kernel_sources':['adithyagiri/name-that-llm-contrastive-v18'],'model_sources':[plan['model']]}
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    out=Path('artifacts/improvement-v21');out.mkdir(exist_ok=True)
    plan['notebook_sha256']=hashlib.sha256((folder/'training.ipynb').read_bytes()).hexdigest()
    (out/'frozen-plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({'folder':str(folder),'notebook_sha256':plan['notebook_sha256']}))


if __name__=='__main__':build()
