"""Build v10: unchanged v7 notebook + v9 Qwen3-4B probe, blended 50/50 on row-standardized evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import nbformat

from build_notebook_v9 import CODE as V9_CODE, C_GRID, LAYERS

V7_NOTEBOOK = Path("notebooks/improvement-v7/name-that-llm-styledistance-v7.ipynb")
V7_SHA256 = "ad5f52f1bebc5a1841d767e96f48a9fc781a8afcdeda934943e919915e719526"

BLEND = r'''
def row_z(s):
    s = np.asarray(s, dtype=float)
    return (s - s.mean(1, keepdims=True)) / s.std(1, keepdims=True)
assert tuple(LABELS) == tuple(sorted(LABELS)) and len(LABELS) == 18
blend = 0.5 * row_z(scores) + 0.5 * row_z(test_scores)
submission = pd.DataFrame({'id': test['id'].to_numpy(), 'model': np.asarray(LABELS)[blend.argmax(axis=1)]})
validate_submission(test, submission)
submission.to_parquet('submission.parquet', index=False)
validate_submission(test, pd.read_parquet('submission.parquet'))
print(json.dumps({'v10_rows': len(submission), 'changed_from_v7': int((blend.argmax(1) != np.asarray(scores).argmax(1)).sum()),
                  'changed_from_v9': int((blend.argmax(1) != test_scores.argmax(1)).sum()),
                  'label_counts': submission['model'].value_counts().to_dict()}, indent=2), flush=True)
'''


def build(folder):
    if hashlib.sha256(V7_NOTEBOOK.read_bytes()).hexdigest() != V7_SHA256:
        raise ValueError("The v7 notebook differs from the Kaggle-verified version")
    nb = nbformat.read(V7_NOTEBOOK, as_version=4)
    nb.cells[0] = nbformat.v4.new_markdown_cell(
        "# Name That LLM — V7 + Qwen3-4B Probe Blend V10\n\n"
        "Runs the unchanged v7 pipeline (StyleDistance probe + v4 stack, CPU), then the v9 frozen "
        "Qwen3-4B-Base hidden-state probe (GPU), and averages row-standardized evidence 50/50. "
        "Out-of-fold: development 40.45% → 45.23%, related-prompt 40.86% → 44.89%, 5/5 fold wins on "
        "each. Both reuse inspected labels and v9's member choice used its own CV. Internet disabled.")
    v9 = V9_CODE.replace("__LAYERS__", repr(LAYERS)).replace("__C_GRID__", repr(C_GRID)).strip()
    nb.cells += [nbformat.v4.new_code_cell(v9), nbformat.v4.new_code_cell(BLEND.strip())]
    nb.metadata["experiment"] = {"name": "improvement-v10", "predecessor_v7_sha256": V7_SHA256}
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<v10 cell>", "exec")
    folder.mkdir(parents=True, exist_ok=True)
    name = "name-that-llm-v7-qwen-blend-v10.ipynb"
    nbformat.write(nb, folder / name)
    meta = json.loads((V7_NOTEBOOK.parent / "kernel-metadata.json").read_text())
    meta.update(id="adithyagiri/name-that-llm-v7-qwen-blend-v10", title="Name That LLM V7 Qwen Blend V10",
                code_file=name, enable_gpu=True, model_sources=["qwen-lm/qwen-3/Transformers/4b-base/1"])
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return folder / name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("notebooks/improvement-v10"))
    print(build(parser.parse_args().output_dir))
