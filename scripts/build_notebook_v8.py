"""Build v8: the unchanged, Kaggle-verified v7 notebook plus within-prompt assignment."""
import argparse
import hashlib
import json
from pathlib import Path

import nbformat

V7_NOTEBOOK = Path("notebooks/improvement-v7/name-that-llm-styledistance-v7.ipynb")
PLAN = Path("configs/name-that-llm/v8.json")
ASSIGNMENT = Path("src/name_that_llm/assignment_v8.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(folder):
    plan = json.loads(PLAN.read_text())
    if sha256(V7_NOTEBOOK) != plan["predecessor_notebook_sha256"]:
        raise ValueError("The v7 notebook differs from the Kaggle-verified version")
    nb = nbformat.read(V7_NOTEBOOK, as_version=4)
    dev, rob = plan["results"]["development"], plan["results"]["robustness"]
    nb.cells[0] = nbformat.v4.new_markdown_cell(
        "# Name That LLM — StyleDistance V7 + Prompt Assignment V8\n\n"
        "V7 evidence is computed unchanged. V8 then recovers prompt groups from test text "
        "(TF-IDF, average-linkage cosine clustering, groups capped at 18) and assigns models "
        "one-to-one within each group by minimum-cost matching, because every training prompt "
        "was answered once by each of the 18 models.\n\n"
        f"Prompt-group development: {dev['v7_argmax']:.4%} → {dev['v8']:.4%}; "
        f"related-prompt check: {rob['v7_argmax']:.4%} → {rob['v8']:.4%}. Both reuse inspected "
        "data, the selection was adaptive, and the gain assumes the hidden test shares the "
        "per-prompt structure. Predictions are transductive.\n\nCPU; internet disabled.")
    source = ASSIGNMENT.read_text()
    nb.cells.append(nbformat.v4.new_code_cell(f'''
(package/'assignment_v8.py').write_text({source!r})
from name_that_llm.assignment_v8 import ASSIGNMENT_PARAMETERS, assign_within_prompts, recover_prompts
v8_started=time.perf_counter()
groups=recover_prompts(test['text'].fillna('').astype(str).tolist())
prediction=assign_within_prompts(scores,groups)
v7_prediction=scores.argmax(axis=1)
submission=pd.DataFrame({{'id':test['id'].to_numpy(),'model':np.asarray(LABELS)[prediction]}})
validate_submission(test,submission)
submission.to_parquet('submission.parquet',index=False)
validate_submission(test,pd.read_parquet('submission.parquet'))
sizes=np.bincount(groups) if groups is not None else np.array([])
v8_audit={{'parameters':ASSIGNMENT_PARAMETERS,'test_rows':len(test),'groups':int(len(sizes)),
          'group_size_counts':{{int(k):int(v) for k,v in zip(*np.unique(sizes,return_counts=True))}},
          'changed_from_v7':int((prediction!=v7_prediction).sum()),
          'assignment_sha256':sha256_file(package/'assignment_v8.py'),
          'seconds':time.perf_counter()-v8_started,
          'label_counts':submission['model'].value_counts().to_dict()}}
audit['v8']=v8_audit
Path('training_audit.json').write_text(json.dumps(audit,indent=2)+'\\n')
print(json.dumps(v8_audit,indent=2),flush=True)
'''.strip()))
    nb.metadata["experiment"] = {"name": "improvement-v8", "predecessor": nb.metadata["experiment"],
                                 "plan_sha256": sha256(PLAN), "assignment_sha256": sha256(ASSIGNMENT)}
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<v8 notebook cell>", "exec")
    folder.mkdir(parents=True, exist_ok=True)
    filename = "name-that-llm-prompt-assignment-v8.ipynb"
    nbformat.write(nb, folder / filename)
    v7_meta = json.loads((V7_NOTEBOOK.parent / "kernel-metadata.json").read_text())
    metadata = {**v7_meta, "id": "adithyagiri/name-that-llm-prompt-assignment-v8",
                "title": "Name That LLM Prompt Assignment V8", "code_file": filename}
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return folder / filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("notebooks/improvement-v8"))
    print(build(parser.parse_args().output_dir))
