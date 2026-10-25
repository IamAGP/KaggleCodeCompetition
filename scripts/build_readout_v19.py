"""Build private paired-checkpoint readout diagnostics on Kaggle."""
import hashlib
import json
from pathlib import Path
import nbformat


def build():
    folder = Path("notebooks/readout-v19");folder.mkdir(exist_ok=True)
    output = Path("artifacts/improvement-v19");output.mkdir(exist_ok=True)
    plan = json.loads(Path("configs/name-that-llm/v19.json").read_text())
    root = Path("artifacts/improvement-v18/kaggle-output")
    paths = {"ce_external":"ce/external_only/adapter", "ce_final":"ce/final/adapter",
             "supcon_external":"ce_supcon/external_only/adapter", "supcon_final":"ce_supcon/final/adapter"}
    plan["adapter_sha256"] = {k:hashlib.sha256((root/p/"adapter_model.safetensors").read_bytes()).hexdigest() for k,p in paths.items()}
    plan["upstream_worker_sha256"] = hashlib.sha256((root/"worker.py").read_bytes()).hexdigest()
    plan["host_sha256"] = hashlib.sha256(Path("data/name-that-llm/train.csv").read_bytes()).hexdigest()
    worker = Path("scripts/readout_v19.py").read_text()
    code = f'''
import os, sys, json, glob, time, subprocess, hashlib, importlib.metadata
from pathlib import Path
from packaging.version import Version
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try: torchao_version = importlib.metadata.version("torchao")
except importlib.metadata.PackageNotFoundError: torchao_version = None
if torchao_version is not None and Version(torchao_version) < Version("0.16.0"):
    subprocess.run([sys.executable,"-m","pip","uninstall","-y","torchao"],check=True)
import torch, peft, transformers
assert torch.cuda.device_count()==2
print({{"torch":torch.__version__,"peft":peft.__version__,"transformers":transformers.__version__}},flush=True)
plan = json.loads({json.dumps(json.dumps(plan))})
matches = [p for p in glob.glob("/kaggle/input/**/plan.json",recursive=True)
           if json.loads(Path(p).read_text()).get("experiment")=="improvement-v18"]
assert len(matches)==1,matches
upstream = Path(matches[0]).parent
assert hashlib.sha256((upstream/"worker.py").read_bytes()).hexdigest()==plan["upstream_worker_sha256"]
plan["upstream_path"] = str(upstream)
paths = [p for p in glob.glob("/kaggle/input/**/config.json",recursive=True) if "/4b-base/" in p.lower()]
assert len(paths)==1,paths
plan["model_dir"] = str(Path(paths[0]).parent)
host = next(p for p in glob.glob("/kaggle/input/**/train.csv",recursive=True) if "name-that-llm" in p)
assert hashlib.sha256(Path(host).read_bytes()).hexdigest()==plan["host_sha256"]
plan["host_path"] = host
Path("plan.json").write_text(json.dumps(plan,indent=2))
Path("worker.py").write_text({worker!r})
assignments = [["base","ce_external","ce_final"],["supcon_external","supcon_final"]]
processes = []
for gpu,variants in enumerate(assignments):
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),V19_VARIANTS=json.dumps(variants))
    processes.append(subprocess.Popen([sys.executable,"-u","worker.py"],env=env))
try:
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None,0) for p in processes):
            raise RuntimeError("Readout worker failed; inspect worker traceback")
        time.sleep(10)
    assert all(p.returncode==0 for p in processes)
except BaseException:
    for p in processes:
        if p.poll() is None:p.terminate()
    for p in processes:p.wait()
    raise
results = {{v:json.loads(Path(v+"_result.json").read_text()) for v in plan["variants"]}}
base = results["base"]["accuracy"]
promoted = [v for v,r in results.items() if v!="base" and r["accuracy"]-base>=0.01 and r["accuracy"]>=0.4044444444]
summary = {{"results":results,"promoted_for_confirmation":promoted,"submission_authorized":False}}
Path("v19_summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
'''
    compile(code,"v19","exec")
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        "# V19: matched readout diagnostic\n\nCompare unadapted Qwen3-4B with our v18 external-only "
        "and final adapters. Fixed three-layer standardized logistic readouts, held-out v18 prompts only. "
        "No submission. Qwen, LoRA and SupCon retain their original credits; external pool credited to "
        "Nishant Dahal and original contributors (CC BY-SA 4.0)."),nbformat.v4.new_code_cell(code)])
    nb.metadata["kernelspec"]={"display_name":"Python 3","language":"python","name":"python3"}
    nbformat.validate(nb);nbformat.write(nb,folder/"readout.ipynb")
    meta={"id":"adithyagiri/name-that-llm-readout-v19","title":"Name That LLM Readout V19",
          "code_file":"readout.ipynb","language":"python","kernel_type":"notebook","is_private":True,
          "enable_gpu":True,"enable_tpu":False,"enable_internet":False,"machine_shape":"NvidiaTeslaT4",
          "competition_sources":["name-that-llm"],"dataset_sources":[],
          "kernel_sources":[plan["upstream_kernel"]],"model_sources":[plan["model"]]}
    (folder/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
    plan["notebook_sha256"]=hashlib.sha256((folder/"readout.ipynb").read_bytes()).hexdigest()
    plan["worker_sha256"]=hashlib.sha256(worker.encode()).hexdigest()
    (output/"frozen-plan.json").write_text(json.dumps(plan,indent=2))
    print(folder)


if __name__=="__main__":build()
