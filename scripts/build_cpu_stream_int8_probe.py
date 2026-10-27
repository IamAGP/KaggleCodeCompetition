"""Distinct bounded streaming-load/INT8 probe, runtime fixture gates real allocation."""
import ast,hashlib,json
from pathlib import Path
import nbformat
root=Path('artifacts/cpu-stream-int8-probe');root.mkdir(exist_ok=True)
old=Path('artifacts/cpu-int8-probe/worker.py').read_text();helper=Path('scripts/cpu_streaming_loader.py').read_text()
start=old.index('def load_trunk(');end=old.index('@torch.no_grad()\ndef merge_lora',start)
worker=old[:start]+helper+'''\n\ndef load_trunk(base_dir,keep,dtype):
    return load_streamed_trunk(base_dir,keep,dtype)


'''+old[end:]
worker=worker.replace('    merge_lora(model, os.path.join(ckpt_dir, "adapter"))','    print(json.dumps({"event":"before_merge",**memory_snapshot()}),flush=True)\n    merge_lora(model, os.path.join(ckpt_dir, "adapter"))\n    print(json.dumps({"event":"after_merge",**memory_snapshot()}),flush=True)',1)
worker=worker.replace("ckpt_name='lora_qwen3_14b';mc=LORAS[ckpt_name]","validate_sharded_fixture(merge_lora)\ngc.collect()\nprint(json.dumps({'event':'before_real_model',**memory_snapshot()}),flush=True)\nckpt_name='lora_qwen3_14b';mc=LORAS[ckpt_name]")
worker=worker.replace("'total':len(names),'rss_bytes':rss","'total':len(names),**memory_snapshot()")
worker=worker.replace("'seconds':elapsed_forward,'rows':len(idx)","'seconds':elapsed_forward,'rows':len(idx),**memory_snapshot()")
compile(worker,'streamworker','exec');(root/'worker.py').write_text(worker)
nb=nbformat.read('artifacts/cpu-int8-probe/probe.ipynb',4)
runner=nb.cells[1].source.replace(repr(old),repr(worker));assert runner!=nb.cells[1].source
runner=runner.replace('        if elapsed>1800 or forward_elapsed>limit or rss>27*2**30:', '''        cg=Path('/sys/fs/cgroup/memory.current')
        cgroup_current=int(cg.read_text()) if cg.exists() else 0
        if elapsed>1800 or forward_elapsed>limit or rss>27*2**30 or cgroup_current>27*2**30:''')
runner=runner.replace("'rss_bytes':rss,'forward_limit':limit","'rss_bytes':rss,'cgroup_current':cgroup_current,'forward_limit':limit")
compile(runner,'streamrunner','exec');nb.cells[1].source=runner;nb.cells[0].source='# Streaming CPU INT8 probe\nMandatory tiny sharded-loader FP16 equality fixture before real14B allocation. Then matchedFP16/INT8 forward, originalLoRA/readout.27GiB RSS and cgroup cap,1800s total. No submission.'
nbformat.validate(nb);nbformat.write(nb,root/'probe.ipynb')
m=json.loads(Path('artifacts/cpu-int8-probe/kernel-metadata.json').read_text());m.update(id='adithyagiri/name-that-llm-cpu-stream-int8-probe',title='Name That LLM CPU Streaming INT8 Probe')
(root/'kernel-metadata.json').write_text(json.dumps(m,indent=2))
a=ast.parse(old);b=ast.parse(worker)
for name in ['merge_lora','find_base']:
 x=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==name);y=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name==name);assert ast.dump(x)==ast.dump(y),name
manifest={'worker_sha256':hashlib.sha256(worker.encode()).hexdigest(),'notebook_sha256':hashlib.sha256((root/'probe.ipynb').read_bytes()).hexdigest(),'loader_sha256':hashlib.sha256(helper.encode()).hexdigest(),'bounds':{'rss_and_cgroup_bytes':27*2**30,'reference_forward_seconds':600,'int8_forward_seconds':300,'whole_seconds':1800},'fixture':'Mandatory runtime same-environment tiny3-layerBF16sharded checkpoint ->2-layerFP16 comparison, allparams/buffers equality before/afterunchangedLoRAmerge andbitwiseforward','original_merge_lora_ast_identical':True,'incumbent_unchanged':True,'sources':['https://huggingface.co/docs/safetensors/en/index','https://huggingface.co/docs/accelerate/en/package_reference/big_modeling','https://huggingface.co/docs/accelerate/en/package_reference/utilities','https://docs.python.org/3/library/os.html#os.posix_fadvise'],'diagnosis':'Standardloader transient29.247GB RSS reached ourguard; notOOM. Tensor-privatecopy/closeperread minimizesmmapretention; memorybenefit requiresmeasurement.'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
