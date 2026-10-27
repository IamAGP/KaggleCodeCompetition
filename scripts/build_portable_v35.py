"""Derive sequential single-GPU V35 without changing data or screening gates."""
import hashlib
import json
from pathlib import Path
import nbformat
from build_training_v34 import replace_once


def build():
    nb=nbformat.read('notebooks/training-v35/training.ipynb',4)
    code=nb.cells[2].source
    code=replace_once(code,'assert torch.cuda.device_count()==2','assert torch.cuda.device_count()==1')
    code=code.replace("Path('/kaggle/input')", "Path(os.environ['V35_INPUT_ROOT'])")
    code=replace_once(code,"plan['deadline_unix']=time.time()+plan['protocol']['runtime']['wall_limit_seconds']",
                      "plan['portable_runtime']=plan['protocol']['runtime']['portable']\nplan['deadline_unix']=time.time()+plan['portable_runtime']['wall_limit_seconds']")
    start=code.index('    for gpu,arm in enumerate(plan[\'arms\']):')
    end=code.index('    assert all(p.returncode==0 for p in processes)',start)
    code=code[:start]+'''    for arm in plan['arms']:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
        process=subprocess.Popen([sys.executable,'-u','run_confirmation_arm_v35.py',arm],env=env,start_new_session=True)
        processes.append(process)
        while process.poll() is None:
            if time.time()>plan['deadline_unix']:raise TimeoutError('V35 portable hard deadline')
            time.sleep(10)
        assert process.returncode==0,arm
'''+code[end:]
    provenance = """import importlib.metadata as metadata, platform
versions={package:metadata.version(package) for package in ['torch','transformers','peft','numpy','pandas','scikit-learn','safetensors','pyarrow','scipy']}
for package,version in {'torch':'2.11.0+cu128','transformers':'5.16.1','peft':'0.20.0'}.items():
    assert versions[package]==version,(package,versions[package],version)
Path('dependency-provenance.json').write_text(json.dumps({'versions':versions,'python':sys.version,'platform':platform.platform()},indent=2))
subprocess.run([sys.executable,'-m','pip','freeze'],check=True,stdout=Path('pip-freeze.txt').open('w'))
import torch
"""
    code=replace_once(code,'import torch\n',provenance)
    compile(code,'portable-v35','exec')
    out=Path('artifacts/improvement-v35/portable')
    out.mkdir(exist_ok=False)
    (out/'run.py').write_text(code)
    manifest={'source_notebook_sha256':hashlib.sha256(Path('notebooks/training-v35/training.ipynb').read_bytes()).hexdigest(),
              'run_sha256':hashlib.sha256(code.encode()).hexdigest(),
              'input_root_environment':'V35_INPUT_ROOT', 'input_requirements':['exact host train.csv','Qwen3-4B base snapshot','V18 ce_supcon/external_only checkpoint'],
              'dependencies':'Verify torch2.11.0+cu128, transformers5.16.1 and peft0.20.0, matching observed V18 loaded log. Standard readout packages also required; setup.py only verifies, never installs.',
              'wall_limit_seconds':9000,'estimated_run_seconds':7000,'observed_train_peak_allocated_gib':8.0445,
              'gpu_requirement':'16GB minimum based on original T4 run; prefer >=24GB for margin. Actual total VRAM exceeds torch allocated peak.',
              'launched':False,'cost_guard':'Parent must enforce $10 total including setup and storage; script watchdog does not terminate cloud billing.'}
    (out/'setup.py').write_text("import importlib.metadata as m\nfor package,version in {'torch':'2.11.0+cu128','transformers':'5.16.1','peft':'0.20.0'}.items():\n    assert m.version(package)==version,(package,m.version(package),version)\nimport pandas,numpy,sklearn,safetensors,pyarrow\nprint('V35 dependency verification passed')\n")
    manifest['setup_sha256']=hashlib.sha256((out/'setup.py').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest))

if __name__=='__main__':build()
