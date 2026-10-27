"""Prepare only: unchanged dense fp16 incumbent on CPU, bounded 3-row probe."""
import ast
import hashlib
import json
from pathlib import Path
import nbformat


def replace_once(s,a,b):
    assert s.count(a)==1,a
    return s.replace(a,b,1)


def build():
    original=Path('notebooks/improvement-v21/inference.ipynb')
    frozen=json.loads(Path('artifacts/improvement-v21/inference-plan.json').read_text())
    assert hashlib.sha256(original.read_bytes()).hexdigest()==frozen['inference_notebook_sha256']
    notebook=nbformat.read(original,4)
    sources=[]
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        s=cell.source
        if 'assert torch.cuda.device_count() >= 2' in s:
            s=s.replace("assert torch.cuda.device_count() >= 2, 'Dual GPUs required'","assert not torch.cuda.is_available(), 'CPU-only probe'")
        if 'def load_trunk(' in s:
            start=s.index('    if gib > 13 and torch.cuda.device_count() > 1:')
            end=s.index('    model.norm = nn.Identity()',start)
            s=s[:start]+'''    model = AutoModel.from_pretrained(base_dir, config=mc, dtype=dtype,
                                      device_map={'':'cpu'}, low_cpu_mem_usage=True).eval()
'''+s[end:]
            s=s.replace('torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16','torch.float16')
        if 'source=' in s and 'component_v21.py' in s:
            tree=ast.parse(s)
            node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='source' for t in n.targets))
            before=ast.literal_eval(node.value)
            after=before.replace('import gc','import contextlib\nimport gc',1).replace('.cuda()',".to('cpu')")
            after=after.replace("with torch.autocast('cuda',dtype=torch.float16):",'with contextlib.nullcontext():')
            s=s.replace(repr(before),repr(after)).replace(hashlib.sha256(before.encode()).hexdigest(),hashlib.sha256(after.encode()).hexdigest())
        s=s.replace('.to("cuda:0")','.to("cpu")')
        s=s.replace("INPUT_DIR = \"/kaggle/input\"","INPUT_DIR = os.environ.get('CPU_PROBE_INPUT_ROOT','/kaggle/input')")
        s=s.replace("glob.glob('/kaggle/input/**/plan.json'","glob.glob(INPUT_DIR+'/**/plan.json'")
        if 'test = pd.read_csv(find("test.csv"))' in s:
            s=s.replace('test = pd.read_csv(find("test.csv"))','test = pd.read_csv(find("test.csv"))\nassert len(test)==3, "This probe is only for the three visible rows; never hidden scoring"')
        s=s.replace('submission.parquet','cpu_probe_predictions.parquet')
        compile(s,'cpu-probe-cell','exec');sources.append(s)
    worker="import os\nos.environ['CUDA_VISIBLE_DEVICES']=''\nos.environ['OMP_NUM_THREADS']='4'\nos.environ['MKL_NUM_THREADS']='4'\n"+'\n\n'.join(sources)
    worker+="\nimport resource\nPath('cpu_probe_runtime.json').write_text(json.dumps({'seconds':time.time()-T0,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rows':len(test),'scope':'3-row probe; not hidden runtime validation'}))\n"
    compile(worker,'cpu-probe-worker','exec')
    root=Path('artifacts/cpu-probe-v21');root.mkdir(exist_ok=True)
    (root/'worker.py').write_text(worker)
    runner="import os,sys,subprocess\nfrom pathlib import Path\n"+f"Path('cpu_probe_worker.py').write_text({worker!r})\n"+"subprocess.run([sys.executable,'-u','cpu_probe_worker.py'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True,timeout=1800)\n"
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# V21 CPU compatibility probe\nThree visible rows only. Dense fp16 remains unchanged; no quantization. 30-minute cap. Not authorized for hidden submission.'),nbformat.v4.new_code_cell(runner)])
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    nbformat.validate(nb);nbformat.write(nb,root/'probe.ipynb')
    metadata=json.loads(Path('notebooks/improvement-v21/kernel-metadata.json').read_text())
    metadata.update(id='adithyagiri/name-that-llm-cpu-compatibility-probe',title='Name That LLM CPU Compatibility Probe',code_file='probe.ipynb',enable_gpu=False)
    metadata.pop('machine_shape',None);(root/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2))
    manifest={'original_notebook_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'worker_sha256':hashlib.sha256(worker.encode()).hexdigest(),'changes':['CPU device placement','Dense fp16 preserved; no bitsandbytes quantization','V21 autocast disabled on CPU while fp16 model weights retained','Require exactly3visible rows','4CPU threads requested','30minute subprocess timeout','Prediction output renamed to diagnostic filename'],'unchanged':['Weights, adapters, training data and SVC specification','Tokenization, pooling and readout','Member blend weights','Topic clustering and centered twins','Sinkhorn postprocessing'],'risk':'CPU fp16 support/performance and device-dependent numerical drift are unmeasured. Current model loading peak memory unknown. Three-row probe cannot establish hidden runtime.','launched':False}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

if __name__=='__main__':build()
