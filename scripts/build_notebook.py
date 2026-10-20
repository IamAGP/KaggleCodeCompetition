"""Embed the exact validated implementation and selected configuration."""

import argparse
import hashlib
import json
from pathlib import Path

import nbformat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/baseline-v1/metrics.json"))
    parser.add_argument("--output", type=Path, default=Path("notebooks/name-that-llm-baseline.ipynb"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    metrics = json.loads(args.metrics.read_text())
    variant = metrics["selected_variant"]
    config = metrics["candidates"][variant]["config"]
    source = (root / "src/name_that_llm/baseline.py").read_text()
    # The reusable definitions are identical; the experiment orchestration is local-only.
    source = source.split("\nclass BaselineExperiment:")[0]
    config_json = json.dumps(config, sort_keys=True)
    runner = f'''import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
config = BaselineConfig(**json.loads({config_json!r}))
override = os.environ.get("NAME_THAT_LLM_DATA_DIR")
candidates = [Path(override)] if override else [
    Path("/kaggle/input/competitions/name-that-llm"),
    Path("/kaggle/input/name-that-llm"),
]
data_dir = next((path for path in candidates
                 if (path / "train.csv").is_file() and (path / "test.csv").is_file()), None)
if data_dir is None:
    raise FileNotFoundError("Attach the name-that-llm competition data.")
train, test = load_data(data_dir)
print(json.dumps({{"versions": runtime_versions(), "config": asdict(config),
                  "train_rows": len(train), "test_rows": len(test)}}, indent=2))
model = fit_checked(build_pipeline(config), train["text"], train["model"])
submission = predict_submission(model, test)
submission.to_parquet("submission.parquet", index=False)
validate_submission(test, pd.read_parquet("submission.parquet"))
print(f"Verified submission.parquet: {{len(submission)}} rows")
'''
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(
                "# Name That LLM — first baseline\n\n"
                "CPU, internet disabled, competition data only. This notebook trains on "
                "all labeled passages and predicts every runtime test row.\n\n"
                f"Chosen variant: `{variant}`. Local five-fold prompt-grouped balanced accuracy: "
                f"**{metrics['candidates'][variant]['pooled_oof_balanced_accuracy']:.4%}**. "
                "This is a development estimate, not a leaderboard score. Related topics "
                "can cross prompt groups; the full dataset informed the earlier qualitative review.\n\n"
                "The three visible test passages check the file format only. Kaggle supplies "
                "the hidden test set during scoring. Output: `submission.parquet`."
            ),
            nbformat.v4.new_code_cell(source),
            nbformat.v4.new_code_cell(runner),
        ],
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "baseline": {
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "train_sha256": metrics["train_sha256"],
                "selected_variant": variant,
            },
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
