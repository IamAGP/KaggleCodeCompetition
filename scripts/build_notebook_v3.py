"""Build an offline notebook only after a frozen v3 recipe passes validation."""

import argparse
import hashlib
import json
from pathlib import Path

import nbformat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, default=Path("artifacts/improvement-v3/selection-second/selected.json"))
    parser.add_argument("--robustness", type=Path, default=Path("artifacts/improvement-v3/robustness"))
    args = parser.parse_args()
    selected = json.loads(args.selected.read_text())
    robustness = json.loads((args.robustness / "metrics.json").read_text())
    frozen = json.loads((args.robustness / "plan.json").read_text())
    if not selected["passes_development_gate"] or not robustness["passes"]:
        raise RuntimeError("The candidate must pass both predeclared validation gates.")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(args.selected) != frozen["selected_sha256"]:
        raise RuntimeError("Selected recipe changed after the frozen comparison.")
    for name, expected in frozen["source_sha256"].items():
        if digest(Path(name)) != expected:
            raise RuntimeError(f"Model source changed after validation: {name}")
    sources = {p.name: p.read_text() for p in sorted(Path("src/name_that_llm").glob("*.py"))}
    source_hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()}
    recipe = json.dumps(selected["recipe"])
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Name That LLM — Validated Ensemble V3\n\n"
            "Text-only classification, CPU, internet disabled, no external data or pretrained weights.\n\n"
            f"Prompt-grouped development balanced accuracy: **{selected['balanced_accuracy']:.4%}**. "
            f"Frozen second-split comparison: **{robustness['v3']:.4%}**, versus "
            f"**{robustness['v2']:.4%}** for v2. These comparisons reuse the available "
            "labeled data and are not independent test estimates.\n\n"
            "The model trains on all labeled passages and predicts each runtime test ID. "
            "Predictions do not depend on other test passages or assumed class counts."
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
import warnings
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from name_that_llm.baseline import load_data, predict_submission, validate_submission, runtime_versions
from name_that_llm.ensemble_v3 import V3Ensemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
recipe = json.loads({recipe!r})
candidates = [Path("/kaggle/input/competitions/name-that-llm"), Path("/kaggle/input/name-that-llm")]
if os.environ.get("NAME_THAT_LLM_DATA_DIR"):
    candidates = [Path(os.environ["NAME_THAT_LLM_DATA_DIR"])]
data_dir = next((p for p in candidates if (p / "train.csv").is_file() and (p / "test.csv").is_file()), None)
if data_dir is None:
    raise FileNotFoundError("Attach the name-that-llm competition data.")
train, test = load_data(data_dir)
versions = runtime_versions()
if any(c.startswith("boosting:") for c in recipe["components"]):
    import lightgbm
    versions["lightgbm"] = lightgbm.__version__
print(json.dumps({{"versions": versions, "recipe": recipe, "train_rows": len(train), "test_rows": len(test)}}, indent=2))
with threadpool_limits(limits=2):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = V3Ensemble(**recipe).fit(train["text"], train["model"], groups=train["prompt_id"])
    for warning in caught:
        logging.warning("%s: %s", warning.category.__name__, warning.message)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("A classifier did not converge; do not submit.")
    submission = predict_submission(model, test)
submission.to_parquet("submission.parquet", index=False)
validate_submission(test, pd.read_parquet("submission.parquet"))
print(f"Verified submission.parquet: {{len(submission)}} rows")
'''.strip()),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "experiment": {"name": "improvement-v3", "embedded_source_sha256": source_hashes,
                       "selected_sha256": digest(args.selected)},
    })
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<notebook cell>", "exec")
    folder = Path("notebooks/improvement-v3")
    folder.mkdir(exist_ok=True)
    nbformat.write(notebook, folder / "name-that-llm-validated-ensemble-v3.ipynb")
    metadata = {
        "id": "adithyagiri/name-that-llm-validated-ensemble-v3",
        "title": "Name That LLM Validated Ensemble V3",
        "code_file": "name-that-llm-validated-ensemble-v3.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "competition_sources": ["name-that-llm"], "dataset_sources": [],
        "kernel_sources": [], "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(folder / metadata["code_file"])


if __name__ == "__main__":
    main()
