"""Package paired GPU training; no public pretrained competition head is used."""
import hashlib
import json
from pathlib import Path
import nbformat
import pandas as pd


def build():
    folder = Path("notebooks/training-v18")
    folder.mkdir(exist_ok=True)
    plan = json.loads(Path("configs/name-that-llm/v18.json").read_text())
    folds = pd.read_csv("artifacts/baseline-v1/folds.csv")
    assert folds.groupby("prompt_id").fold.nunique().max() == 1
    mapping = folds.groupby("prompt_id").fold.first().to_dict()
    for key, path in [("host_sha256", "data/name-that-llm/train.csv"),
                      ("pool_sha256", "data/name-that-llm/external/pool_v4.parquet"),
                      ("folds_sha256", "artifacts/baseline-v1/folds.csv")]:
        plan[key] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    worker = Path("scripts/train_adapter_v18.py").read_text()
    prep = Path("scripts/prepare_v18.py").read_text()
    code = f'''
import os, sys, json, glob, time, subprocess
from pathlib import Path
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch, peft, transformers
assert torch.cuda.device_count() == 2, "Paired pilot requires two GPUs"
print({{"torch":torch.__version__, "peft":peft.__version__, "transformers":transformers.__version__}}, flush=True)
plan = json.loads({json.dumps(json.dumps(plan))})
fold_by_prompt = json.loads({json.dumps(json.dumps(mapping))})
Path("worker.py").write_text({worker!r})
Path("prepare_data.py").write_text({prep!r})
from prepare_data import prepare
host_paths = glob.glob("/kaggle/input/**/train.csv", recursive=True)
host = next(p for p in host_paths if "name-that-llm" in p)
pool_paths = glob.glob("/kaggle/input/**/pool_v4.parquet", recursive=True)
assert len(pool_paths) == 1
model_paths = [p for p in glob.glob("/kaggle/input/**/config.json", recursive=True) if "/4b-base/" in p.lower()]
assert len(model_paths) == 1
plan["model_dir"] = str(Path(model_paths[0]).parent)
plan["labels"] = prepare(host, pool_paths[0], fold_by_prompt, plan, ".")
Path("plan.json").write_text(json.dumps(plan, indent=2))
processes = []
for gpu, variant in enumerate(plan["variants"]):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), V18_VARIANT=variant)
    # Inherit stdout/stderr so Kaggle records actual worker progress and exceptions.
    processes.append((variant, subprocess.Popen([sys.executable, "-u", "worker.py"], env=env)))
try:
    while any(p.poll() is None for _,p in processes):
        failed = [(v,p.returncode) for v,p in processes if p.poll() not in (None,0)]
        if failed:
            raise RuntimeError(f"Paired worker failed: {{failed}}")
        time.sleep(10)
    assert all(p.returncode == 0 for _,p in processes)
except BaseException:
    for _,p in processes:
        if p.poll() is None: p.terminate()
    for _,p in processes: p.wait()
    raise
results = {{v:json.loads(Path(v,"result.json").read_text()) for v,_ in processes}}
gain = results["ce_supcon"]["accuracy"] - results["ce"]["accuracy"]
summary = {{"results":results, "gain":gain, "promote":gain >= 0.01, "submission_authorized":False}}
Path("v18_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
'''
    compile(code, "v18", "exec")
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        "# V18: cross-prompt author contrastive training\n\nPrivate paired-objective pilot. "
        "LoRA: Hu et al. (2106.09685); SupCon: Khosla et al. (2004.11362), "
        "https://github.com/HobbitLong/SupContrast. Our joint objective is an adaptation, not a paper reproduction. "
        "External pool: Nishant Dahal and original contributors, CC BY-SA 4.0. "
        "No competition-fine-tuned checkpoint is used; host validation prompts are excluded. "
        "No submission output is produced."), nbformat.v4.new_code_cell(code)])
    nb.metadata["kernelspec"] = {"display_name":"Python 3", "language":"python", "name":"python3"}
    nbformat.validate(nb)
    nbformat.write(nb, folder / "training.ipynb")
    meta = {"id":"adithyagiri/name-that-llm-contrastive-v18", "title":"Name That LLM Contrastive V18",
            "code_file":"training.ipynb", "language":"python", "kernel_type":"notebook",
            "is_private":True, "enable_gpu":True, "enable_tpu":False, "enable_internet":False,
            "machine_shape":"NvidiaTeslaT4", "competition_sources":["name-that-llm"],
            "dataset_sources":["nishantdahal/ntl-handwriting-blend"], "kernel_sources":[],
            "model_sources":[plan["model"]]}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    out = Path("artifacts/improvement-v18"); out.mkdir(exist_ok=True)
    plan["notebook_sha256"] = hashlib.sha256((folder / "training.ipynb").read_bytes()).hexdigest()
    plan["worker_sha256"] = hashlib.sha256(worker.encode()).hexdigest()
    (out / "frozen-plan.json").write_text(json.dumps(plan, indent=2))
    print(folder)


if __name__ == "__main__":
    build()
