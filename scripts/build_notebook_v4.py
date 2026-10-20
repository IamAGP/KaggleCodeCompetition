"""Build the offline nested-stack notebook after both frozen gates pass."""

import argparse
import hashlib
import json
from pathlib import Path

import nbformat


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(selected_path, robustness_path, folder):
    selected = json.loads(selected_path.read_text())
    robustness = json.loads(robustness_path.read_text())
    if not selected["passes_development_gate"] or not robustness["passes"]:
        raise ValueError("Both predeclared validation gates must pass.")
    if robustness["selected_sha256"] != digest(selected_path):
        raise ValueError("The selected recipe changed after robustness validation.")
    if selected["recipe"] != robustness["recipe"]:
        raise ValueError("Development and robustness recipes differ.")
    if selected["source_hashes"] != robustness["source_hashes"]:
        raise ValueError("Model source changed between evaluations.")
    for key, path in {
        "selector_sha256": Path("scripts/select_v4.py"),
        "evaluator_sha256": Path("scripts/run_nested_v4.py"),
        "plan_sha256": Path("configs/name-that-llm/v4.json"),
    }.items():
        if selected[key] != robustness[key] or selected[key] != digest(path):
            raise ValueError(f"Validated experiment provenance changed: {key}")
    for name, expected in robustness["source_hashes"].items():
        if digest(Path("src/name_that_llm") / name) != expected:
            raise ValueError(f"Validated source changed: {name}")
    for result in (selected, robustness):
        for path, expected in result["input_sha256"].items():
            if digest(path) != expected:
                raise ValueError(f"Validation evidence changed: {path}")
    recipe = json.dumps(selected["recipe"])
    sources = {p.name: p.read_text() for p in sorted(Path("src/name_that_llm").glob("*.py"))}
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Name That LLM — Nested Stack V4\n\n"
            "Four text/style classifiers feed a regularized multinomial logistic combiner. "
            "The combiner is trained on predictions made without training on each passage's prompt group.\n\n"
            f"Nested development balanced accuracy: **{selected['balanced_accuracy']:.4%}**. "
            f"Frozen related-prompt comparison: **{robustness['balanced_accuracy']:.4%}**, "
            f"versus **{robustness['baseline']['balanced_accuracy']:.4%}** for v3. "
            "Available labeled data is reused; these are not independent test estimates.\n\n"
            "CPU, internet disabled, no external data or pretrained weights. "
            "Test IDs retain their original order; predictions are independent of other test passages."
        ),
        nbformat.v4.new_code_cell(
            'import os\n'
            'for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):\n'
            '    os.environ[key] = "2"\n'
        ),
        nbformat.v4.new_code_cell(
            "import json\nfrom pathlib import Path\nimport sys\n"
            f"sources = json.loads({json.dumps(sources)!r})\n"
            "package = Path.cwd() / 'name_that_llm'\n"
            "package.mkdir(exist_ok=True)\n"
            "for name, source in sources.items():\n"
            "    (package / name).write_text(source)\n"
            "sys.path.insert(0, str(package.parent))\n"
        ),
        nbformat.v4.new_code_cell(f'''
import logging
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from name_that_llm.baseline import LABELS, load_data, validate_submission, runtime_versions
from name_that_llm.ensemble_v2 import normalize_evidence
from name_that_llm.stacking_v4 import NestedStacker, evidence_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
recipe = json.loads({recipe!r})
candidates = [Path("/kaggle/input/competitions/name-that-llm"), Path("/kaggle/input/name-that-llm")]
if os.environ.get("NAME_THAT_LLM_DATA_DIR"):
    candidates = [Path(os.environ["NAME_THAT_LLM_DATA_DIR"])]
data_dir = next((p for p in candidates if (p / "train.csv").is_file() and (p / "test.csv").is_file()), None)
if data_dir is None:
    raise FileNotFoundError("Attach the name-that-llm competition data.")
train, test = load_data(data_dir)
print(json.dumps({{"versions": runtime_versions(), "recipe": recipe,
                  "train_rows": len(train), "test_rows": len(test)}}, indent=2))
with threadpool_limits(limits=2):
    model = NestedStacker(threads=2).fit(train["text"], train["model"], groups=train["prompt_id"])
    base = model.base_evidence(test["text"])
    stack = normalize_evidence(model.meta_.predict_proba(evidence_features(base)), probabilities=True)
    scores = recipe["stack_weight"] * stack + recipe["v3_weight"] * base[:, [0, 1, 3], :].mean(axis=1)
    submission = pd.DataFrame({{"id": test["id"].to_numpy(), "model": np.asarray(LABELS)[scores.argmax(axis=1)]}})
validate_submission(test, submission)
submission.to_parquet("submission.parquet", index=False)
validate_submission(test, pd.read_parquet("submission.parquet"))
Path("training_audit.json").write_text(json.dumps(model.fit_oof_audit_, indent=2) + "\\n")
print(f"Verified submission.parquet: {{len(submission)}} rows")
'''.strip()),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "experiment": {
            "name": "improvement-v4", "selected_sha256": digest(selected_path),
            "robustness_sha256": digest(robustness_path),
            "embedded_source_sha256": {
                name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()
            },
        },
    })
    nbformat.validate(notebook)
    for cell in cells:
        if cell.cell_type == "code":
            compile(cell.source, "<v4 notebook cell>", "exec")
    folder.mkdir(parents=True, exist_ok=True)
    filename = "name-that-llm-nested-stack-v4.ipynb"
    nbformat.write(notebook, folder / filename)
    metadata = {
        "id": "adithyagiri/name-that-llm-nested-stack-v4",
        "title": "Name That LLM Nested Stack V4", "code_file": filename,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "competition_sources": ["name-that-llm"], "dataset_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return folder / filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, default=Path("artifacts/improvement-v4/selection/selected.json"))
    parser.add_argument("--robustness", type=Path, default=Path("artifacts/improvement-v4/robustness/check/metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("notebooks/improvement-v4"))
    args = parser.parse_args()
    print(build(args.selected, args.robustness, args.output_dir))


if __name__ == "__main__":
    main()
