"""Prepare full-host 8B CE only after the unchanged related V34 gate passes."""
import hashlib
import json
from pathlib import Path
from build_training_v34 import replace_once


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    related=Path('artifacts/improvement-v34/related')
    result=related/'evaluation-audit.json'
    assert result.exists(), 'Related V34 has not been audited; no full-host package authorized'
    audit=json.loads(result.read_text())
    assert audit['verified'] and audit['qualifies_for_full_host_training'], 'Related gate failed'
    frozen=json.loads((related/'frozen-plan.json').read_text())
    assert audit['protocol_sha256']==frozen['protocol_sha256']
    spec=frozen['arms']['8b_ce']
    worker=Path('scripts/train_adapter_v25.py')
    assert sha(worker)==spec['origin_worker_sha256']
    training=Path('scripts/train_full_v26.py').read_text()
    training=replace_once(training,"assert plan['experiment'] == 'improvement-v26'","assert plan['experiment'] == 'improvement-v34-full'")
    training=replace_once(training,"assert plan['pilot_qualification']['verified']","assert plan['related_qualification']['verified'] and plan['related_qualification']['qualifies_for_full_host_training']")
    training=replace_once(training,"selected = plan['pilot_qualification']['selected_for_full_host_training']","selected = 'ce_final'")
    component=Path('scripts/component_v26.py').read_text()
    component=replace_once(component,"assert plan['experiment']=='improvement-v26' and plan['pilot_qualification']['verified']",
        "assert plan['experiment']=='improvement-v34-full' and plan['related_qualification']['verified'] and plan['related_qualification']['qualifies_for_full_host_training']")
    sources={'v18_worker.py':worker.read_text(),'train_full.py':training,'component_v34.py':component}
    for name,source in sources.items():compile(source,name,'exec')
    params=dict(spec['parameters']);params['contrastive_weight']=0.0
    plan={'experiment':'improvement-v34-full','training_parameters':params,'labels':frozen['labels'],
        'host_sha256':frozen['host_sha256'],'checkpoint_sha256':spec['checkpoint_sha256'],
        'readout':spec['readout'],'related_qualification':audit,'related_audit_sha256':sha(result),
        'protocol_sha256':frozen['protocol_sha256'],'submission_authorized':False,
        'source_sha256':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}}
    code='''import os,sys,json,hashlib,subprocess,time
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
'''
    code+=f'plan=json.loads({json.dumps(json.dumps(plan))})\nsources={sources!r}\n'
    code+='''assert not Path('full').exists(),'Never overwrite full-host outputs'
root=Path(os.environ['V34_INPUT_ROOT'])
host=list(root.rglob('train.csv'));assert len(host)==1
assert hashlib.sha256(host[0].read_bytes()).hexdigest()==plan['host_sha256']
plan['host_path']=str(host[0])
models=[]
for p in root.rglob('config.json'):
    c=json.loads(p.read_text())
    if c.get('model_type')=='qwen3' and c.get('hidden_size')==4096 and c.get('num_hidden_layers')==36:models.append(p.parent)
assert len(models)==1
plan['model_dir']=str(models[0])
weights=[p for p in root.rglob('adapter_model.safetensors') if p.parent.parent.name=='external_only' and hashlib.sha256(p.read_bytes()).hexdigest()==plan['checkpoint_sha256']['adapter/adapter_model.safetensors']]
assert len(weights)==1
plan['checkpoint']=str(weights[0].parent.parent)
for name,source in sources.items():
    assert hashlib.sha256(source.encode()).hexdigest()==plan['source_sha256'][name]
    Path(name).write_text(source)
Path('plan.json').write_text(json.dumps(plan,indent=2))
deadline=time.monotonic()+9000
for script in ['train_full.py','component_v34.py']:
    subprocess.run([sys.executable,'-u',script],check=True,timeout=max(1,deadline-time.monotonic()),env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
assert json.loads(Path('full/training_result.json').read_text())['completed']
assert Path('full/readout.npz').is_file()
print('Full-host component complete; no submission created')
'''
    compile(code,'v34-full','exec')
    out=Path('artifacts/improvement-v34/full-host');out.mkdir(exist_ok=False)
    (out/'run.py').write_text(code)
    plan['run_sha256']=sha(out/'run.py')
    (out/'frozen-plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({'folder':str(out),'launched':False,'run_sha256':plan['run_sha256']}))


if __name__=='__main__':build()
