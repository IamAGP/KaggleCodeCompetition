"""Build a self-contained offline notebook for the frozen v2 recipe."""

import hashlib
import json
from pathlib import Path
import re

import nbformat


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "artifacts/improvement-v2"
    selected = json.loads((run / "selection/selected.json").read_text())
    robustness = json.loads((run / "robustness/metrics.json").read_text())
    if not robustness["passes"]:
        raise RuntimeError("The frozen candidate did not pass its robustness comparison.")
    recipe_json = json.dumps(selected["recipe"])
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Name That LLM — Style Ensemble V2\n\n"
            "Text-only, CPU, internet disabled, no external data. "
            "Character patterns and fixed writing-style measurements supply "
            "complementary evidence for each passage.\n\n"
            f"Development balanced accuracy: **{selected['balanced_accuracy']:.4%}**. "
            f"Second split with related prompts grouped: **{robustness['improved']:.4%}**, "
            f"compared with **{robustness['baseline']:.4%}** for the original baseline. "
            "Both use the available labeled development data and are not independent test scores.\n\n"
            "The notebook fits on all labeled rows and generates one prediction per runtime test ID. "
            "It does not assume equal test class counts or fixed test size."
        ),
        nbformat.v4.new_code_cell(
            'import os\n'
            'for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):\n'
            '    os.environ[variable] = "2"\n'
        ),
    ]
    hashes = {}
    for module in ("baseline", "sparse_v2", "stylometry", "generative", "ensemble_v2"):
        source = (root / f"src/name_that_llm/{module}.py").read_text()
        if module == "baseline":
            source = source.split("\nclass BaselineExperiment:")[0]
        # The exact same definitions live in notebook cells, so local-package
        # imports are unnecessary. All third-party imports remain unchanged.
        source = re.sub(r"^from name_that_llm\.[^\n]+\n", "", source, flags=re.MULTILINE)
        hashes[module] = hashlib.sha256(source.encode()).hexdigest()
        cells.append(nbformat.v4.new_code_cell(source))
    cells.append(nbformat.v4.new_code_cell(f'''logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
recipe = json.loads({recipe_json!r})
candidates = [
    Path("/kaggle/input/competitions/name-that-llm"),
    Path("/kaggle/input/name-that-llm"),
]
override = os.environ.get("NAME_THAT_LLM_DATA_DIR")
if override:
    candidates = [Path(override)]
data_dir = next((path for path in candidates if (path / "train.csv").is_file() and (path / "test.csv").is_file()), None)
if data_dir is None:
    raise FileNotFoundError("Attach the name-that-llm competition data.")
train, test = load_data(data_dir)
print(json.dumps({{"versions": runtime_versions(), "recipe": recipe,
                  "train_rows": len(train), "test_rows": len(test)}}, indent=2))
model = EvidenceEnsemble(**recipe).fit(train["text"], train["model"])
submission = predict_submission(model, test)
submission.to_parquet("submission.parquet", index=False)
validate_submission(test, pd.read_parquet("submission.parquet"))
print(f"Verified submission.parquet: {{len(submission)}} rows")
'''))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "experiment": {"name": "improvement-v2", "embedded_source_sha256": hashes},
    })
    nbformat.validate(notebook)
    folder = root / "notebooks/improvement-v2"
    folder.mkdir(exist_ok=True)
    nbformat.write(notebook, folder / "name-that-llm-style-ensemble-v2.ipynb")
    metadata = {
        "id": "adithyagiri/name-that-llm-style-ensemble-v2",
        "title": "Name That LLM Style Ensemble V2",
        "code_file": "name-that-llm-style-ensemble-v2.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "competition_sources": ["name-that-llm"],
        "dataset_sources": [], "kernel_sources": [], "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(folder / "name-that-llm-style-ensemble-v2.ipynb")


if __name__ == "__main__":
    main()
