"""Prepare FP16-reference vs FBGEMM INT8 numerical-variant CPU probe."""
import hashlib,json
from pathlib import Path
import nbformat
root=Path('artifacts/cpu-int8-probe');root.mkdir(exist_ok=True)
src=Path('artifacts/cpu-bf16-probe/worker.py').read_text()
src=src.replace('dtype = torch.bfloat16','dtype = torch.float16')
src=src.replace("FORWARD_TIMES=[]",'''FORWARD_TIMES=[]
PHASE='fp16_reference'
engines=list(torch.backends.quantized.supported_engines)
assert 'fbgemm' in engines, engines
torch.backends.quantized.engine='fbgemm'
smoke=nn.Linear(32,16).float().eval()
smoke.qconfig=torch.ao.quantization.per_channel_dynamic_qconfig
smokeq=torch.ao.nn.quantized.dynamic.Linear.from_float(smoke)
smokeout=smokeq(torch.ones(2,32,dtype=torch.float32))
assert smokeout.dtype==torch.float32 and torch.isfinite(smokeout).all()
print(json.dumps({'event':'int8_smoke_pass','engines':engines,'engine':torch.backends.quantized.engine,'output_dtype':str(smokeout.dtype)}),flush=True)
del smoke,smokeq,smokeout

def current_rss():
    return int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024

def convert_linears(model):
    targets=[(name,module) for name,module in model.named_modules() if isinstance(module,nn.Linear)]
    # Do not keep references to all original modules while converting: that would retain their weights.
    names=[name for name,module in targets];del targets
    for i,name in enumerate(names):
        parent_path,_,key=name.rpartition('.')
        parent=model.get_submodule(parent_path) if parent_path else model
        module=getattr(parent,key)
        assert isinstance(module,nn.Linear)
        module.float()  # This layer only; global model.float before conversion exceeds RAM.
        module.qconfig=torch.ao.quantization.per_channel_dynamic_qconfig
        quantized=torch.ao.nn.quantized.dynamic.Linear.from_float(module)
        setattr(parent,key,quantized)
        del module,quantized;gc.collect()
        rss=current_rss()
        assert rss<27*2**30, f'RSS ceiling exceeded: {rss}'
        if i%14==0 or i+1==len(names):
            print(json.dumps({'event':'int8_conversion','converted':i+1,'total':len(names),'rss_bytes':rss}),flush=True)
    # Remaining embedding/norm/residual/attention arithmetic is FP32; packed INT8 linears have no floating Parameter weights.
    model.float()
    assert model.embed_tokens.weight.dtype==torch.float32
    return model
''')
src=src.replace("{'started':started_forward,'rows':len(idx)","{'started':started_forward,'limit':600 if PHASE=='fp16_reference' else 300,'phase':PHASE,'rows':len(idx)")
src=src.replace("{'event':'forward_start','rows'","{'event':'forward_start','phase':PHASE,'rows'")
src=src.replace("{'event':'forward_complete','seconds'","{'event':'forward_complete','phase':PHASE,'seconds'")
src=src.replace('FORWARD_TIMES.append(elapsed_forward)',"FORWARD_TIMES.append({'phase':PHASE,'seconds':elapsed_forward})")
start=src.index('        if not all(torch.isfinite(hs[l]).all() for l in layers):')
end=src.index('        m = enc[',start)
src=src[:start]+"        assert all(torch.isfinite(hs[l]).all() for l in layers), 'Nonfinite activations; no dtype fallback in paired experiment'\n"+src[end:]
src=src.replace('    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))','    global PHASE\n    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))',1)
start=src.index('    for s in range(0, len(order), bs):')
end=src.index('    del model',start)
src=src[:start]+'''    for s in range(0,len(order),bs):embed(order[s:s+bs])
    reference=Z.copy()
    reference_logits=reference@heads['plain.weight'].float().numpy().T+heads['plain.bias'].float().numpy()
    np.savez_compressed('fp16_reference.npz',ids=test.id.to_numpy(dtype=str),labels=np.asarray(LABELS),features=reference,logits=reference_logits)
    PHASE='int8_dynamic'
    model=convert_linears(model)
    Z.fill(0)
    for s in range(0,len(order),bs):embed(order[s:s+bs])
'''+src[end:]
src=src.replace('bf16_member_scores.npz','int8_member_scores.npz').replace("'dtype':'bfloat16'","'dtype':'dynamic_int8_linears_fp32_activations'")
src+='''
with np.load('fp16_reference.npz',allow_pickle=False) as ref:
    delta=np.abs(ref['logits']-scores)
    comparison={'rows':len(test),'max_absolute_logit_delta':float(delta.max()),'mean_absolute_logit_delta':float(delta.mean()),'argmax_agreement':float((ref['logits'].argmax(1)==scores.argmax(1)).mean()),'scope':'Same-machine FP16 vs INT8 first14B member only, no ensemble or hidden-score equivalence'}
Path('parity.json').write_text(json.dumps(comparison,indent=2));print(json.dumps({'event':'parity',**comparison}),flush=True)
'''
compile(src,'int8worker','exec');(root/'worker.py').write_text(src)
runner='''import os,sys,time,json,subprocess,signal
from pathlib import Path
'''+f"Path('worker.py').write_text({src!r})\n"+'''
start=time.monotonic()
p=subprocess.Popen([sys.executable,'-u','worker.py'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
try:
    while p.poll() is None:
        elapsed=time.monotonic()-start
        active=Path('forward-active.json');forward_elapsed=0;limit=600;phase='setup'
        if active.exists():
            try:
                state=json.loads(active.read_text());forward_elapsed=time.monotonic()-state['started'];limit=state['limit'];phase=state['phase']
            except (json.JSONDecodeError,FileNotFoundError):pass
        rss=0
        try:rss=int(next(line.split()[1] for line in Path(f'/proc/{p.pid}/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024
        except (FileNotFoundError,StopIteration):pass
        if elapsed>1800 or forward_elapsed>limit or rss>27*2**30:
            reason={'event':'bounded_stop','whole_seconds':elapsed,'forward_seconds':forward_elapsed,'phase':phase,'rss_bytes':rss,'forward_limit':limit}
            Path('bounded-stop.json').write_text(json.dumps(reason));print(json.dumps(reason),flush=True)
            raise TimeoutError(str(reason))
        time.sleep(0.5)
    if p.returncode:raise RuntimeError(f'worker exited {p.returncode}')
finally:
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
'''
compile(runner,'int8runner','exec')
nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# CPU dynamic INT8 single-member probe\nFP16 reference followed by one-layer-at-a-time FBGEMM conversion. Numerical variant, not incumbent replacement. Three visible rows; no submission.'),nbformat.v4.new_code_cell(runner)],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
nbformat.validate(nb);nbformat.write(nb,root/'probe.ipynb')
m=json.loads(Path('artifacts/cpu-bf16-probe/kernel-metadata.json').read_text());m.update(id='adithyagiri/name-that-llm-cpu-int8-probe',title='Name That LLM CPU INT8 Probe')
(root/'kernel-metadata.json').write_text(json.dumps(m,indent=2))
manifest={'worker_sha256':hashlib.sha256(src.encode()).hexdigest(),'notebook_sha256':hashlib.sha256((root/'probe.ipynb').read_bytes()).hexdigest(),'bounds':{'fp16_forward_seconds':600,'int8_forward_seconds':300,'whole_seconds':1800,'rss_bytes':27*2**30},'dtype_handling':'Load and merge original FP16; reference forward inFP16; eachLinear casts toFP32 for observer+packing then is immediately replaced by dynamicper-channelINT8. RemainingnonlinearsconvertedFP32afteralllinears; floatingactivations dynamicallyquantized insideFBGEMM and floatoutputs; pooling/head remainoriginalFP32.','comparison':'SameprocessmatchingFP16firstmemberlogits, notaggregateensemble','launched':False}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
