"""Package the fixed source-likelihood feasibility probe as an offline notebook."""
import hashlib
import json
from pathlib import Path
import nbformat


def main():
    plan=json.loads(Path('configs/name-that-llm/v31.json').read_text())
    sources={name:Path('scripts',name).read_text() for name in ['source_likelihood.py','probe_likelihood_v31.py']}
    plan['source_sha256']={name:hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()}
    code='from pathlib import Path\nimport subprocess, sys, json\n'
    code+=f"Path('plan.json').write_text({json.dumps(json.dumps(plan,indent=2))})\n"
    for name,source in sources.items():code+=f'Path({name!r}).write_text({source!r})\n'
    code+="subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', 'probe_likelihood_v31.py'], check=True)\n"
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        '# Source-conditioned likelihood feasibility V31\n'
        'Method: Ko et al., ATLAS, https://arxiv.org/abs/2610.19583. '
        'Independent implementation; probe checkpoint must be discarded. No competition submission.'),
        nbformat.v4.new_code_cell(code)],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    folder=Path('notebooks/probe-v31');folder.mkdir(parents=True,exist_ok=True)
    nbformat.write(notebook,folder/'probe.ipynb')
    metadata={'id':'adithyagiri/name-that-llm-likelihood-probe-v31','title':'Name That LLM Likelihood Probe V31',
              'code_file':'probe.ipynb','language':'python','kernel_type':'notebook','is_private':True,
              'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'machine_shape':'NvidiaTeslaT4',
              'competition_sources':['name-that-llm'],'dataset_sources':[],'kernel_sources':[],
              'model_sources':[plan['model']]}
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    plan['notebook_sha256']=hashlib.sha256((folder/'probe.ipynb').read_bytes()).hexdigest()
    root=Path('artifacts/improvement-v31');root.mkdir(exist_ok=True)
    (root/'frozen-plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({'notebook_sha256':plan['notebook_sha256'],'source_sha256':plan['source_sha256']},indent=2))


if __name__=='__main__':main()
