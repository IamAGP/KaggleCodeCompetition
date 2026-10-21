"""Build an offline StyleDistance submission only after both frozen gates pass."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from name_that_llm.classifier_v7 import source_hashes
from name_that_llm.encoder_v7 import sha256_file
from run_embeddings_v7 import validate_plan, verify_selection


def build(folder):
    plan_path = Path("configs/name-that-llm/v7.json")
    plan = validate_plan(plan_path)
    selected_path = Path("artifacts/improvement-v7/development/selected.json")
    robust_path = Path("artifacts/improvement-v7/robustness/metrics.json")
    selected = verify_selection(selected_path)
    if not selected["passes_development_gate"] or selected["delta_from_v4"] < plan["development"]["min_gain"]:
        raise ValueError("The frozen development gate must pass")
    robust = json.loads(robust_path.read_text())
    if (not selected["passes_development_gate"] or selected["delta_from_v4"] < plan["development"]["min_gain"]
            or not robust["passes"] or robust["delta_from_v4"] < plan["robustness"]["min_gain"]
            or robust["fold_wins"] < plan["robustness"]["min_fold_wins"]):
        raise ValueError("Both frozen validation gates must pass")
    if robust["selected_sha256"] != sha256_file(selected_path) or selected["recipe"] != robust["recipe"]:
        raise ValueError("The validated recipe changed")
    completed = json.loads((robust_path.parent / "completion.json").read_text())
    if not all(completed[k] for k in ("all_checks_passed", "source_unchanged_during_run", "inputs_unchanged_during_run")) or completed["result_sha256"] != sha256_file(robust_path):
        raise ValueError("Robustness evaluation did not complete unchanged")
    for entry in json.loads((robust_path.parent / "artifacts.json").read_text())["files"]:
        path = robust_path.parent / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Robustness artifact changed: {path}")
    for result in (selected, robust):
        expected = {"source_hashes": source_hashes(), "plan_sha256": sha256_file(plan_path),
                    "evaluator_sha256": sha256_file(Path(__file__).with_name("run_embeddings_v7.py"))}
        for key, value in expected.items():
            if result[key] != value:
                raise ValueError(f"Validated provenance changed: {key}")
        for path, digest in result["input_sha256"].items():
            if sha256_file(path) != digest:
                raise ValueError(f"Validation input changed: {path}")
    model_manifest = Path("artifacts/improvement-v7/model/manifest.json")
    golden_path = Path("artifacts/improvement-v7/encoder/runtime-golden.json")
    golden = json.loads(golden_path.read_text())
    if golden["provenance"]["manifest_sha256"] != sha256_file(model_manifest):
        raise ValueError("Runtime golden fixture belongs to a different model package")
    if golden["provenance"]["encoder_source_sha256"] != sha256_file("src/name_that_llm/encoder_v7.py"):
        raise ValueError("Runtime golden fixture belongs to different encoder source")
    sources = {p.name: p.read_text() for p in sorted(Path("src/name_that_llm").glob("*.py"))}
    cells = [nbformat.v4.new_markdown_cell(
        "# Name That LLM — StyleDistance V7\n\n"
        "Frozen public StyleDistance representations feed a fixed logistic classifier. "
        + ("The selected recipe averages its evidence with the validated v4 text/style stack. "
           if selected["recipe"]["v4_weight"] else "The selected recipe uses this classifier alone. ")
        +
        f"Prompt-group development: {selected['balanced_accuracy']:.4%}; related-prompt check: {robust['balanced_accuracy']:.4%}. "
        "Both reuse development data and do not estimate an untouched test set.\n\n"
        "CPU; internet disabled; local safetensors verified before loading. Model: StyleDistance/styledistance, "
        "revision b7df5f0b0480773c097ba3121d83ca32b71015ca (MIT). "
        "Paper: https://arxiv.org/abs/2410.12757v2. Each test passage is predicted independently."),
        nbformat.v4.new_code_cell(
            "import os\n"
            "for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT'):\n    os.environ[key]='2'\n"
            "for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_TELEMETRY','HF_HUB_DISABLE_IMPLICIT_TOKEN'):\n    os.environ[key]='1'\n"
            "os.environ['TOKENIZERS_PARALLELISM']='false'\n"),
        nbformat.v4.new_code_cell(
            "import json\nfrom pathlib import Path\nimport sys\n"
            f"sources = json.loads({json.dumps(sources)!r})\n"
            "package=Path.cwd()/'name_that_llm'\npackage.mkdir(exist_ok=True)\n"
            "for name, source in sources.items():\n    (package/name).write_text(source)\n"
            "sys.path.insert(0,str(package.parent))\n"
            f"recipe=json.loads({json.dumps(selected['recipe'])!r})\n"
            f"golden=json.loads({json.dumps(golden)!r})\n"
            f"expected_model_manifest_sha256={sha256_file(model_manifest)!r}\n"),
        nbformat.v4.new_code_cell('''
import gc
import logging
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from name_that_llm.baseline import LABELS, load_data, validate_submission, runtime_versions
from name_that_llm.encoder_v7 import FrozenStyleDistanceEncoder, sha256_file
from name_that_llm.classifier_v7 import fit_classifier, model_evidence, combine_evidence
from name_that_llm.classifier_v5 import incumbent_evidence
from name_that_llm.stacking_v4 import NestedStacker, evidence_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
started=time.perf_counter()
data_paths=[Path('/kaggle/input/competitions/name-that-llm'),Path('/kaggle/input/name-that-llm')]
if os.environ.get('NAME_THAT_LLM_DATA_DIR'):
    data_paths=[Path(os.environ['NAME_THAT_LLM_DATA_DIR'])]
data_dir=next((p for p in data_paths if (p/'train.csv').is_file() and (p/'test.csv').is_file()),None)
if data_dir is None:
    raise FileNotFoundError('Attach the competition data')
manifest_candidates=list(Path('/kaggle/input').rglob('manifest.json'))
if os.environ.get('NAME_THAT_LLM_MODEL_DIR'):
    manifest_candidates=[Path(os.environ['NAME_THAT_LLM_MODEL_DIR'])/'manifest.json']
matching=[p for p in manifest_candidates if sha256_file(p)==expected_model_manifest_sha256]
if len(matching)!=1:
    raise ValueError('Attach exactly one verified StyleDistance model package')
train,test=load_data(data_dir)
encoder=FrozenStyleDistanceEncoder(matching[0].parent)
golden_vectors=encoder.encode(golden['texts'])
golden_error=float(np.max(np.abs(golden_vectors-np.asarray(golden['embeddings']))))
if golden_error>golden['cross_runtime_max_abs_tolerance']:
    raise ValueError(f'Cross-runtime encoder difference exceeds tolerance: {golden_error}')
print(json.dumps({'versions':runtime_versions(),'encoder_runtime':encoder.runtime,
                  'golden_max_abs_difference':golden_error,'train_rows':len(train),'test_rows':len(test)},indent=2))
def progress(record):
    if record['batch']%50==0:
        print(json.dumps(record),flush=True)
train_vectors=encoder.encode(train['text'].tolist(),progress=progress)
train_report=encoder.report
test_vectors=encoder.encode(test['text'].tolist(),progress=progress)
encoder_audit={'provenance':encoder.provenance(),'train':train_report,'test':encoder.report,'golden_error':golden_error}
del encoder
gc.collect()
with threadpool_limits(limits=2):
    linear=fit_classifier(train_vectors,train['model'])
    embedding_scores=model_evidence(linear,test_vectors)
    stacker=None
    old=np.zeros_like(embedding_scores)
    if recipe['v4_weight']:
        stacker=NestedStacker(threads=2).fit(train['text'],train['model'],groups=train['prompt_id'])
        base=stacker.base_evidence(test['text'])
        old=incumbent_evidence(stacker.meta_.predict_proba(evidence_features(base)),base)
    scores=combine_evidence(embedding_scores,old,recipe)
    submission=pd.DataFrame({'id':test['id'].to_numpy(),'model':np.asarray(LABELS)[scores.argmax(axis=1)]})
validate_submission(test,submission)
submission.to_parquet('submission.parquet',index=False)
validate_submission(test,pd.read_parquet('submission.parquet'))
audit={'encoder':encoder_audit,'stacker':stacker.fit_oof_audit_ if stacker is not None else None,'recipe':recipe,
       'runtime_source_sha256':{name:sha256_file(package/name) for name in sources},
       'seconds':time.perf_counter()-started,'submission_rows':len(submission)}
Path('training_audit.json').write_text(json.dumps(audit,indent=2)+'\\n')
print(json.dumps({'verified_submission_rows':len(submission),'seconds':audit['seconds']}))
'''.strip())]
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "experiment": {
            "name": "improvement-v7", "selected_sha256": sha256_file(selected_path),
            "robustness_sha256": sha256_file(robust_path), "golden_sha256": sha256_file(golden_path),
            "embedded_source_sha256": {n: hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}}})
    nbformat.validate(nb)
    for cell in cells:
        if cell.cell_type=="code":
            compile(cell.source,"<v7 notebook cell>","exec")
    folder.mkdir(parents=True,exist_ok=True)
    filename="name-that-llm-styledistance-v7.ipynb"
    nbformat.write(nb,folder/filename)
    metadata={"id":"adithyagiri/name-that-llm-styledistance-v7","title":"Name That LLM StyleDistance V7",
              "code_file":filename,"language":"python","kernel_type":"notebook","is_private":True,
              "enable_gpu":False,"enable_tpu":False,"enable_internet":False,
              "competition_sources":["name-that-llm"],
              "dataset_sources":["adithyagiri/name-that-llm-styledistance-b7df5f0"],"kernel_sources":[],"model_sources":[]}
    (folder/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+'\n')
    return folder/filename


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("notebooks/improvement-v7"))
    print(build(parser.parse_args().output_dir))
