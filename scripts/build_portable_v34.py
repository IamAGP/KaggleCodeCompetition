"""Package the frozen related V34 recipes for sequential single-GPU execution."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

import nbformat
from build_training_v34 import replace_once


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    source = Path('notebooks/training-v34-related/training.ipynb')
    frozen = json.loads(Path('artifacts/improvement-v34/related/frozen-plan.json').read_text())
    assert sha(source) == frozen['notebook_sha256']
    nb = nbformat.read(source, 4)
    code = nb.cells[2].source
    code = replace_once(code, 'assert torch.cuda.device_count()==2', 'assert torch.cuda.device_count()==1')
    assert code.count("Path('/kaggle/input')") == 3
    code = code.replace("Path('/kaggle/input')", "Path(os.environ['V34_INPUT_ROOT'])")
    code = replace_once(code,
        "plan['deadline_unix']=time.time()+plan['protocol']['runtime']['wall_limit_seconds']",
        "plan['portable_runtime']={'single_gpu_sequential':True,'wall_limit_seconds':9000}\nplan['deadline_unix']=time.time()+9000")
    start = code.index("    for gpu,arm in enumerate(plan['arms']):")
    end = code.index('    assert all(p.returncode==0 for p in processes)', start)
    code = code[:start] + '''    for arm in plan['arms']:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
        process=subprocess.Popen([sys.executable,'-u','run_confirmation_arm_v34.py',arm],env=env,start_new_session=True)
        processes.append(process)
        while process.poll() is None:
            if time.time()>plan['deadline_unix']:raise TimeoutError('V34 portable wall budget exhausted')
            time.sleep(10)
        assert process.returncode==0,arm
''' + code[end:]
    compile(code, 'portable-v34', 'exec')
    # Embedded scientific workers must remain byte-identical to the frozen related package.
    embedded = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'write_text':
            target = node.func.value
            if isinstance(target, ast.Call) and isinstance(target.func, ast.Name) and target.func.id == 'Path':
                if len(target.args)==1 and isinstance(target.args[0],ast.Constant) and len(node.args)==1 and isinstance(node.args[0],ast.Constant):
                    embedded[target.args[0].value] = node.args[0].value
    for name,digest in frozen['source_sha256'].items():
        assert hashlib.sha256(embedded[name].encode()).hexdigest()==digest
    out=Path('artifacts/improvement-v34/related/portable')
    out.mkdir(exist_ok=False)
    (out/'run.py').write_text(code)
    setup='''import importlib.metadata as m, json, hashlib, os
from pathlib import Path
for package,version in {'torch':'2.11.0+cu128','transformers':'5.16.1','peft':'0.20.0','bitsandbytes':'0.50.0'}.items():
    assert m.version(package)==version,(package,m.version(package),version)
import pandas,numpy,sklearn,safetensors,pyarrow,accelerate,torch
assert torch.cuda.device_count()==1
assert torch.cuda.get_device_properties(0).total_memory>=16*1024**3
if 'torchao' in {d.metadata['Name'] for d in m.distributions()}:
    from packaging.version import Version
    assert Version(m.version('torchao'))>=Version('0.16.0')
root=Path(os.environ['V34_INPUT_ROOT'])
manifest=json.loads(Path(os.environ['V34_INPUT_MANIFEST']).read_text())
assert manifest['model_sources']==['qwen-lm/qwen-3/Transformers/4b-base/1','qwen-lm/qwen-3/Transformers/8b-base/1']
assert manifest['model_snapshot_receipts_verified'] is True
for entry in manifest['files']:
    p=root/entry['path']
    assert p.is_file() and p.stat().st_size==entry['bytes'],str(p)
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==entry['sha256'],str(p)
versions={p:m.version(p) for p in ['torch','transformers','peft','bitsandbytes','pandas','numpy','scikit-learn','safetensors','pyarrow','accelerate']}
Path('environment.json').write_text(json.dumps({'packages':versions,'gpu':torch.cuda.get_device_name(0)},indent=2))
print('V34 verified inputs and runtime dependencies')
'''
    (out/'setup.py').write_text(setup)
    inputs=out/'inputs'; inputs.mkdir()
    copies=[(Path('data/name-that-llm/train.csv'),inputs/'train.csv')]
    for arm,spec in frozen['arms'].items():
        origin=Path(f"artifacts/improvement-v{spec['origin_version']}/kaggle-output")/spec['variant']/'external_only'
        for relative,digest in spec['checkpoint_sha256'].items():
            assert sha(origin/relative)==digest
            copies.append((origin/relative,inputs/arm/spec['variant']/'external_only'/relative))
    wheel=Path('artifacts/improvement-v24/offline-wheels/bitsandbytes-0.50.0-py3-none-manylinux_2_24_x86_64.whl')
    assert sha(wheel)=='173d137610468bec9cddbaa2e049254e97792657ab984e3e737bec1772c1668c'
    copies.append((wheel,inputs/'wheels'/wheel.name))
    for src,dst in copies:
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    manifest={'model_sources':['qwen-lm/qwen-3/Transformers/4b-base/1','qwen-lm/qwen-3/Transformers/8b-base/1'],
        'model_snapshot_receipts_verified':False,
        'files':[{'path':str(dst.relative_to(inputs)),'bytes':dst.stat().st_size,'sha256':sha(dst)} for _,dst in copies]}
    (out/'input-manifest.partial.json').write_text(json.dumps(manifest,indent=2))
    result={'source_notebook_sha256':sha(source),'run_sha256':sha(out/'run.py'),'setup_sha256':sha(out/'setup.py'),
        'protocol_sha256':frozen['protocol_sha256'],'scientific_worker_hashes_unchanged':True,
        'input_root_environment':'V34_INPUT_ROOT','input_manifest_environment':'V34_INPUT_MANIFEST',
        'input_bytes_without_backbones':sum(f['bytes'] for f in manifest['files']),
        'model_sources':manifest['model_sources'],'wall_limit_seconds':9000,
        'estimated_seconds_t4_sequential':6500,'a40_timing':'Not measured; retain conservative T4 projection',
        'original_core_packages':{'torch':'2.11.0+cu128','transformers':'5.16.1','peft':'0.20.0','bitsandbytes':'0.50.0'},
        'other_dependency_versions':'Original Kaggle log did not capture all versions. Runtime records them; do not claim fully identical environment.',
        'launch_ready':False,'missing':'Exact pinned backbone snapshots and completed hash/size manifest; GPU package installation and setup validation',
        'spend_guard':'9000-second process deadline does not terminate billing; parent must enforce hard $10 total external watchdog.'}
    (out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':build()
