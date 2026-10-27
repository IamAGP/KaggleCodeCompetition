"""Separate CPU BF16 single-member delivery feasibility experiment; no submission."""
import hashlib,json
from pathlib import Path
import nbformat

root=Path('artifacts/cpu-bf16-probe');root.mkdir(exist_ok=True)
src=Path('artifacts/cpu-probe-v21/worker.py').read_text()
prefix=src[:src.index('from scipy.sparse import hstack')]
functions=src[src.index('from safetensors.torch import load_file'):src.index('from scipy.cluster.hierarchy import fcluster, linkage')]
assert functions.count('dtype = torch.float16')==1
functions=functions.replace('dtype = torch.float16','dtype = torch.bfloat16')
needle='        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states'
assert functions.count(needle)==2
functions=functions.replace(needle,'''        started_forward=time.monotonic()
        Path('forward-active.json').write_text(json.dumps({'started':started_forward,'rows':len(idx),'padded_tokens':int(enc['input_ids'].shape[1])}))
        print(json.dumps({'event':'forward_start','rows':len(idx),'padded_tokens':int(enc['input_ids'].shape[1])}),flush=True)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        elapsed_forward=time.monotonic()-started_forward
        Path('forward-active.json').unlink()
        FORWARD_TIMES.append(elapsed_forward)
        print(json.dumps({'event':'forward_complete','seconds':elapsed_forward,'rows':len(idx)}),flush=True)''',1)
hardware='''
import platform, resource
cpuinfo=Path('/proc/cpuinfo').read_text()
flags=next((line.split(':',1)[1].strip().split() for line in cpuinfo.splitlines() if line.startswith('flags')),[])
model_name=next((line.split(':',1)[1].strip() for line in cpuinfo.splitlines() if line.startswith('model name')),None)
hardware={'cpu_model':model_name,'flags':flags,'native_bf16_advertised':bool(set(flags)&{'avx512_bf16','amx_bf16'}),'cpu_capability':torch.backends.cpu.get_cpu_capability(),'torch':torch.__version__,'mkldnn_available':torch.backends.mkldnn.is_available(),'threads':torch.get_num_threads(),'logical_cpus':os.cpu_count()}
Path('hardware.json').write_text(json.dumps(hardware,indent=2));print(json.dumps({'event':'hardware',**hardware}),flush=True)
FORWARD_TIMES=[]
'''
ending='''
ckpt_name='lora_qwen3_14b';mc=LORAS[ckpt_name]
ckpt=os.path.dirname(find(f'{ckpt_name}/meta.json'))
Z,heads=lora_features(ckpt,find_base(mc['base']),test.text.tolist(),mc['bs'])
scores=Z@heads['plain.weight'].T+heads['plain.bias']
assert np.isfinite(scores).all()
np.savez_compressed('bf16_member_scores.npz',ids=test.id.to_numpy(dtype=str),labels=np.asarray(LABELS),features=Z,logits=scores,probabilities=softmax(scores))
result={'event':'probe_complete','dtype':'bfloat16','member':ckpt_name,'rows':len(test),'seconds':time.time()-T0,'forward_seconds':FORWARD_TIMES,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'Three visible rows, single member; no fullensemble parity or hidden runtime claim'}
Path('runtime.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
'''
worker=prefix+hardware+functions+ending
compile(worker,'bf16worker','exec');(root/'worker.py').write_text(worker)
runner='''import os,sys,time,json,subprocess,signal
from pathlib import Path
'''+f"Path('worker.py').write_text({worker!r})\n"+'''
start=time.monotonic()
p=subprocess.Popen([sys.executable,'-u','worker.py'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),start_new_session=True)
try:
    while p.poll() is None:
        elapsed=time.monotonic()-start
        active=Path('forward-active.json')
        forward_elapsed=0
        if active.exists():
            try:forward_elapsed=time.monotonic()-json.loads(active.read_text())['started']
            except (json.JSONDecodeError,FileNotFoundError):pass
        if elapsed>1800 or forward_elapsed>300:
            reason={'event':'bounded_timeout','whole_seconds':elapsed,'forward_seconds':forward_elapsed}
            Path('timeout.json').write_text(json.dumps(reason));print(json.dumps(reason),flush=True)
            raise TimeoutError(str(reason))
        time.sleep(0.5)
    if p.returncode:raise RuntimeError(f'worker exited {p.returncode}')
finally:
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
'''
compile(runner,'bf16runner','exec')
nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# CPU BF16 single-member probe\nSame public checkpoint, truncation, adapter and pooling; distinct dtype experiment. Three visible rows only; no submission.'),nbformat.v4.new_code_cell(runner)],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
nbformat.validate(nb);nbformat.write(nb,root/'probe.ipynb')
m=json.loads(Path('artifacts/cpu-probe-v21/kernel-metadata.json').read_text())
m.update(id='adithyagiri/name-that-llm-cpu-bf16-probe',title='Name That LLM CPU BF16 Probe',model_sources=['qwen-lm/qwen-3/Transformers/14b-base/1'],kernel_sources=[])
(root/'kernel-metadata.json').write_text(json.dumps(m,indent=2))
manifest={'source_worker_sha256':hashlib.sha256(src.encode()).hexdigest(),'worker_sha256':hashlib.sha256(worker.encode()).hexdigest(),'notebook_sha256':hashlib.sha256((root/'probe.ipynb').read_bytes()).hexdigest(),'dtype':'bfloat16','bounds':{'forward_seconds':300,'whole_seconds':1800},'preserved':'checkpoint, truncation, LoRA merge, norms, tokenization, pooling, plain readout; only dtype plus diagnostics changed; SVC/other members omitted','sources':['https://pytorch.org/blog/empowering-pytorch-on-intel-xeon-scalable-processors-with-bfloat16/','https://docs.pytorch.org/docs/main/backends.html','https://github.com/pytorch/pytorch/blob/main/docs/source/amp.md'],'hypothesis':'Native AVX512_BF16/AMX if advertised may outperform CPUfp16; hardware and speed remain unmeasured until run. BF16 changes numerical rounding and requires independent parity evaluation.'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
