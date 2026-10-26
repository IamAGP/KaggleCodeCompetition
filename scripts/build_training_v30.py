"""Package the fixed likelihood pilot only after the independent GPU audit."""
import hashlib
import json
from pathlib import Path
import nbformat
import pandas as pd


def main():
    plan=json.loads(Path('configs/name-that-llm/v30.json').read_text())
    audit_path=Path('artifacts/improvement-v31/probe-audit-v2.json')
    audit=json.loads(audit_path.read_text())
    assert audit['verified'] and audit['qualifies_for_pilot']
    assert audit['source_sha256']['source_likelihood.py']==hashlib.sha256(Path('scripts/source_likelihood.py').read_bytes()).hexdigest()
    plan['feasibility_audit']=audit
    plan['feasibility_audit_sha256']=hashlib.sha256(audit_path.read_bytes()).hexdigest()
    assert hashlib.sha256(Path('artifacts/baseline-v1/folds.csv').read_bytes()).hexdigest()==plan['folds_sha256']
    folds=pd.read_csv('artifacts/baseline-v1/folds.csv',dtype={'prompt_id':str})
    assert folds.groupby('prompt_id').fold.nunique().max()==1
    mapping=folds.groupby('prompt_id').fold.first().to_dict()
    files=['source_likelihood.py','likelihood_batches.py','train_likelihood_v30.py','prepare_v18.py']
    sources={name:Path('scripts',name).read_text() for name in files}
    plan['source_sha256']={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}
    code="import os, sys, subprocess, json\nfrom pathlib import Path\n"
    code+="for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[name]='2'\n"
    code+="os.environ['TOKENIZERS_PARALLELISM']='false'\n"
    for name,source in sources.items():code+=f'Path({name!r}).write_text({source!r})\n'
    code+=f'plan=json.loads({json.dumps(json.dumps(plan))})\n'
    code+=f'fold_by_prompt=json.loads({json.dumps(json.dumps(mapping))})\n'
    code+='''
from prepare_v18 import prepare
hosts=list(Path('/kaggle/input').rglob('train.csv'))
pools=list(Path('/kaggle/input').rglob('pool_v4.parquet'))
assert len(hosts)==len(pools)==1
models=[]
for path in Path('/kaggle/input').rglob('config.json'):
    cfg=json.loads(path.read_text())
    if cfg.get('model_type')=='qwen2' and cfg.get('hidden_size')==1536 and cfg.get('num_hidden_layers')==28:
        models.append(path.parent)
assert len(models)==1
plan['model_dir']=str(models[0])
plan['labels']=prepare(hosts[0],pools[0],fold_by_prompt,plan,'.')
Path('plan.json').write_text(json.dumps(plan,indent=2))
subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                'train_likelihood_v30.py'],check=True)
print(json.dumps({'training_complete':True,'decision':'Independent local fixed-gate evaluation required',
                  'submission_authorized':False}),flush=True)
'''
    compile(code,'v30','exec')
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        '# Source-conditioned likelihood pilot V30\n'
        'ATLAS method: Ko et al., https://arxiv.org/abs/2610.19583. Independent implementation. '
        'Public pool: Nishant Dahal and original contributors, CC BY-SA 4.0. '
        'Fresh Qwen2.5-1.5B base; no feasibility checkpoint or host validation labels used in training. '
        'Fixed final epoch; no submission.'),nbformat.v4.new_code_cell(code)],
        metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    folder=Path('notebooks/training-v30');folder.mkdir(parents=True,exist_ok=True)
    nbformat.write(notebook,folder/'training.ipynb')
    metadata={'id':'adithyagiri/name-that-llm-likelihood-training-v30','title':'Name That LLM Likelihood Training V30',
              'code_file':'training.ipynb','language':'python','kernel_type':'notebook','is_private':True,
              'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4',
              'competition_sources':['name-that-llm'],'dataset_sources':['nishantdahal/ntl-handwriting-blend'],
              'kernel_sources':[],'model_sources':[plan['model']]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    plan['notebook_sha256']=hashlib.sha256((folder/'training.ipynb').read_bytes()).hexdigest()
    root=Path('artifacts/improvement-v30');root.mkdir(exist_ok=True)
    (root/'frozen-plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({'folder':str(folder),'notebook_sha256':plan['notebook_sha256']}))


if __name__=='__main__':main()
