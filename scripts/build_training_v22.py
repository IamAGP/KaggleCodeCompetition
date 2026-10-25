"""Scale the fixed v18 training data, then append unchanged v19 readout workers."""
import hashlib
import json
from pathlib import Path
import nbformat


def build():
    # Reuse the proven packaging code without modifying its frozen files.
    template=Path('scripts/build_training_v18.py').read_text()
    replacements={
        'notebooks/training-v18':'notebooks/training-v22',
        'configs/name-that-llm/v18.json':'configs/name-that-llm/v22.json',
        'artifacts/improvement-v18':'artifacts/improvement-v22',
        'adithyagiri/name-that-llm-contrastive-v18':'adithyagiri/name-that-llm-scaled-training-v22',
        'Name That LLM Contrastive V18':'Name That LLM Scaled Training V22',
        'v18_summary.json':'v22_head_diagnostic.json',
    }
    for old,new in replacements.items():
        assert old in template,old
        template=template.replace(old,new)
    # The inherited head-only promotion calculation is not the v22 decision.
    old='summary = {{"results":results, "gain":gain, "promote":gain >= 0.01, "submission_authorized":False}}'
    new='summary = {{"results":results, "gain":gain, "kind":"learned-head diagnostic only", "submission_authorized":False}}'
    assert old in template;template=template.replace(old,new)
    namespace={'__name__':'v22_packaging'}
    exec(compile(template,'v18-packaging-reused-for-v22','exec'),namespace)
    namespace['build']()
    folder=Path('notebooks/training-v22');root=Path('artifacts/improvement-v22')
    nb=nbformat.read(folder/'training.ipynb',4)
    worker=Path('scripts/readout_v19.py').read_text()
    readout=json.loads(Path('configs/name-that-llm/v22.json').read_text())['readout']
    extra=f'''
import hashlib
training_plan=json.loads(Path('plan.json').read_text())
Path('training-plan.json').write_text(json.dumps(training_plan,indent=2))
readout_plan=dict(training_plan)
readout_plan.update({readout!r})
readout_plan['upstream_path']='.'
readout_plan['host_path']=host
readout_plan['adapter_sha256']={{v:hashlib.sha256(Path(p).read_bytes()).hexdigest() for v,p in {{
    'ce_final':'ce/final/adapter/adapter_model.safetensors',
    'supcon_final':'ce_supcon/final/adapter/adapter_model.safetensors'}}.items()}}
Path('readout-plan.json').write_text(json.dumps(readout_plan,indent=2))
Path('plan.json').write_text(json.dumps(readout_plan,indent=2))
Path('readout_worker.py').write_text({worker!r})
processes=[]
for gpu,variant in enumerate(['ce_final','supcon_final']):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),V19_VARIANTS=json.dumps([variant]))
    processes.append(subprocess.Popen([sys.executable,'-u','readout_worker.py'],env=env))
try:
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('Scaled readout failed')
        time.sleep(10)
    assert all(p.returncode==0 for p in processes)
except BaseException:
    for p in processes:
        if p.poll() is None:p.terminate()
    for p in processes:p.wait()
    raise
finally:
    Path('plan.json').write_text(json.dumps(training_plan,indent=2))
summary={{'experiment':'improvement-v22','readouts':{{v:json.loads(Path(v+'_result.json').read_text())
         for v in ['ce_final','supcon_final']}},'decision':'Requires local fixed-weight sparse blend check',
         'submission_authorized':False}}
Path('v22_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
'''
    compile(extra,'v22-readout','exec')
    nb.cells[0]=nbformat.v4.new_markdown_cell(
        '# V22: external data scaling\n\nMatched CE and CE+SupCon arms repeat the v18 method with '
        '1,000 external passages per label (18,000 total) instead of 250. One pool-equivalent epoch '
        'and two host-equivalent epochs; same held-out prompts, overlap screen and optimizer. '
        'Append fixed v19 readouts at layers 12/24/36. No submission. '
        'LoRA: Hu et al., arXiv:2106.09685; SupCon: Khosla et al., arXiv:2004.11362 '
        'and github.com/HobbitLong/SupContrast. Our joint attribution objective is an adaptation. '
        'Public pool: Nishant Dahal and original contributors, CC BY-SA 4.0.')
    nb.cells.append(nbformat.v4.new_code_cell(extra))
    nbformat.validate(nb);nbformat.write(nb,folder/'training.ipynb')
    plan=json.loads((root/'frozen-plan.json').read_text())
    plan['notebook_sha256']=hashlib.sha256((folder/'training.ipynb').read_bytes()).hexdigest()
    plan['readout_worker_sha256']=hashlib.sha256(worker.encode()).hexdigest()
    plan['packaging_template_sha256']=hashlib.sha256(Path('scripts/build_training_v18.py').read_bytes()).hexdigest()
    (root/'frozen-plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({'folder':str(folder),'notebook_sha256':plan['notebook_sha256']}))


if __name__=='__main__':build()
