"""Package a conditional confirmation split; never launch or bypass V30's gate."""
import argparse
import hashlib
import json
from pathlib import Path

import nbformat
import pandas as pd


SPLITS = ['primary-1', 'primary-2', 'primary-3', 'primary-4', 'related-0']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_qualified_audit(path):
    path = Path(path)
    if not path.exists():
        raise RuntimeError('V30 completion audit is not available; confirmation remains unlaunched')
    audit = json.loads(path.read_text())
    if audit.get('verified') is not True or audit.get('qualifies_for_broader_validation') is not True:
        raise RuntimeError('V30 did not pass the frozen confirmation prerequisite')
    return audit


def build(split):
    assert split in SPLITS
    root = Path('artifacts/improvement-v30')
    audit_path = root / 'evaluation-audit.json'
    audit = require_qualified_audit(audit_path)
    out = root / 'kaggle-output'
    assert sha(out / 'external-only-adapter.safetensors') == audit['checkpoint_sha256']['external-only']
    frozen = json.loads((root / 'frozen-plan.json').read_text())
    plan = json.loads((out / 'plan.json').read_text())
    for key, value in frozen.items():
        if key != 'notebook_sha256':
            assert plan[key] == value, key
    for name, expected in plan['source_sha256'].items():
        assert sha(out / name) == expected
    for name in ['source_likelihood.py', 'likelihood_batches.py']:
        assert sha(Path('scripts') / name) == plan['source_sha256'][name]
    protocol = json.loads(Path('configs/name-that-llm/v33.json').read_text())
    family, fold = split.split('-')
    fold = int(fold)
    fold_key = family + '_fold_file'
    fold_path = Path(protocol[fold_key])
    assert sha(fold_path) == protocol['fold_sha256'][fold_key]
    frame = pd.read_csv(fold_path, dtype={'id': str, 'prompt_id': str, 'group': str})
    host_path = Path('data/name-that-llm/train.csv')
    assert sha(host_path) == plan['host_sha256']
    host = pd.read_csv(host_path, dtype={'id': str, 'prompt_id': str})
    assert frame.id.equals(host.id) and frame.prompt_id.equals(host.prompt_id)
    if 'group' not in frame:
        frame['group'] = 'host:' + frame.prompt_id
    assert frame.groupby('prompt_id').fold.nunique().max() == 1
    assert frame.groupby('prompt_id').group.nunique().max() == 1
    assert frame.groupby('group').fold.nunique().max() == 1
    mapping = frame.groupby('prompt_id')[['group', 'fold']].first().to_dict(orient='index')
    # Every scale change is logged by V30, and scale growth cannot occur in this run.
    assert plan['loss_scale_growth_interval'] > 1825
    scales = []
    for rank in range(2):
        log = [json.loads(line) for line in (out / f'progress-rank{rank}.jsonl').read_text().splitlines()]
        pool_updates = [e for e in log if e['event'] == 'update' and e['stage'] == 'pool']
        assert log[-1]['event'] == 'complete' and pool_updates
        scales.append(pool_updates[-1]['scale'])
    assert scales[0] == scales[1] and 0 < scales[0] <= plan['initial_loss_scale']
    plan.update(experiment='improvement-v33', confirmation_split=split, confirmation_fold=fold,
                training_rows=int((frame.fold != fold).sum()), validation_rows=int((frame.fold == fold).sum()),
                v30_audit=audit, v30_audit_sha256=sha(audit_path), confirmation_protocol=protocol,
                external_checkpoint_loss_scale=scales[0], parent_notebook_sha256=frozen['notebook_sha256'])
    plan.pop('model_dir', None)
    files = ['source_likelihood.py', 'likelihood_batches.py', 'prepare_likelihood_v33.py', 'train_likelihood_v33.py']
    sources = {name: Path('scripts', name).read_text() for name in files}
    plan['source_sha256'] = {name: hashlib.sha256(s.encode()).hexdigest() for name, s in sources.items()}
    code = "import os, sys, subprocess, json, hashlib\nfrom pathlib import Path\n"
    code += "for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='2'\n"
    code += "os.environ['TOKENIZERS_PARALLELISM']='false'\n"
    for name, source in sources.items():
        code += f'Path({name!r}).write_text({source!r})\n'
    code += f'plan=json.loads({json.dumps(json.dumps(plan))})\n'
    code += f'splits=json.loads({json.dumps(json.dumps(mapping))})\n'
    code += '''
from prepare_likelihood_v33 import prepare
hosts=list(Path('/kaggle/input').rglob('train.csv'))
assert len(hosts)==1
models=[]
for path in Path('/kaggle/input').rglob('config.json'):
    cfg=json.loads(path.read_text())
    if cfg.get('model_type')=='qwen2' and cfg.get('hidden_size')==1536 and cfg.get('num_hidden_layers')==28:
        models.append(path.parent)
assert len(models)==1
checkpoints=[p for p in Path('/kaggle/input').rglob('external-only-adapter.safetensors')
             if hashlib.sha256(p.read_bytes()).hexdigest()==plan['v30_audit']['checkpoint_sha256']['external-only']]
assert len(checkpoints)==1
plan['external_checkpoint']=str(checkpoints[0])
plan['model_dir']=str(models[0])
prepare(hosts[0],splits,plan,'.')
Path('plan.json').write_text(json.dumps(plan,indent=2))
subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                'train_likelihood_v33.py'],check=True)
print(json.dumps({'complete':True,'submission_authorized':False,'split':plan['confirmation_split']}),flush=True)
'''
    compile(code, 'v33', 'exec')
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        '# Source likelihood confirmation V33: ' + split + '\n'
        'ATLAS method: Ko et al., https://arxiv.org/abs/2610.19583. Independent implementation. '
        'External-only initialization from V30, using the public pool by Nishant Dahal and original '
        'contributors (CC BY-SA 4.0). Qwen2.5 by Qwen (Apache 2.0). Fixed host holdout; no submission.'),
        nbformat.v4.new_code_cell(code)],
        metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    folder = Path('notebooks') / ('training-v33-' + split)
    assert not folder.exists(), 'An existing confirmation package must not be overwritten'
    folder.mkdir(parents=True)
    nbformat.write(notebook, folder / 'training.ipynb')
    metadata = {'id': 'adithyagiri/name-that-llm-likelihood-v33-' + split,
                'title': 'Name That LLM Likelihood V33 ' + split, 'code_file': 'training.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
                'machine_shape': 'NvidiaTeslaT4', 'competition_sources': ['name-that-llm'],
                'dataset_sources': [], 'kernel_sources': ['adithyagiri/name-that-llm-likelihood-training-v30'],
                'model_sources': [plan['model']]}
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    plan['notebook_sha256'] = sha(folder / 'training.ipynb')
    dest = Path('artifacts/improvement-v33') / split
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'frozen-plan.json').write_text(json.dumps(plan, indent=2))
    print(json.dumps({'folder': str(folder), 'launched': False, 'notebook_sha256': plan['notebook_sha256']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=SPLITS, required=True)
    build(parser.parse_args().split)
