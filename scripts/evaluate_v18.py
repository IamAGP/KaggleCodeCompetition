"""Audit completed paired training outputs against the frozen host validation fold."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def evaluate(folder):
    folder = Path(folder)
    plan = json.loads((folder / "plan.json").read_text())
    host_path = Path("data/name-that-llm/train.csv")
    assert hashlib.sha256(host_path.read_bytes()).hexdigest() == plan["host_sha256"]
    host = pd.read_csv(host_path, dtype={"id":str, "prompt_id":str})
    folds_path = Path("artifacts/baseline-v1/folds.csv")
    assert hashlib.sha256(folds_path.read_bytes()).hexdigest() == plan["folds_sha256"]
    folds = pd.read_csv(folds_path, dtype={"id":str, "prompt_id":str})
    assert folds.id.equals(host.id) and folds.prompt_id.equals(host.prompt_id)
    valid = host.loc[folds.fold == plan["pilot_fold"]].reset_index(drop=True)
    labels = np.asarray(plan["labels"])
    expected = valid.model.map({label:i for i,label in enumerate(labels)}).to_numpy()
    correct = {}
    predictions = {}
    summaries = {}
    hashes = {}
    for variant in plan["variants"]:
        path = folder / variant / "validation.npz"
        result = json.loads((folder / variant / "result.json").read_text())
        assert result["completed"] and result["variant"] == variant
        with np.load(path, allow_pickle=False) as data:
            assert np.array_equal(data["ids"], valid.id.to_numpy())
            assert np.array_equal(data["labels"], labels)
            assert np.array_equal(data["targets"], expected)
            logits = data["logits"]
            assert logits.shape == (len(valid),18) and np.isfinite(logits).all()
            prediction = logits.argmax(1)
        predictions[variant] = prediction
        correct[variant] = prediction == expected
        accuracy = float(correct[variant].mean())
        assert abs(accuracy-result["accuracy"]) < 1e-10
        summaries[variant] = {"accuracy":accuracy, "skipped_steps":result["skipped_steps"], "seconds":result["seconds"]}
        hashes[variant] = hashlib.sha256(path.read_bytes()).hexdigest()
    differences = correct["ce_supcon"].astype(float)-correct["ce"].astype(float)
    grouped = pd.DataFrame({"prompt":valid.prompt_id, "difference":differences}).groupby("prompt").difference.agg(["sum","size"])
    rng = np.random.default_rng(1801)
    samples = rng.integers(len(grouped), size=(10000,len(grouped)))
    bootstrap = grouped["sum"].to_numpy()[samples].sum(1)/grouped["size"].to_numpy()[samples].sum(1)
    gain = float(differences.mean())
    audit = {"results":summaries, "gain":gain,
             "corrected_errors":int((differences==1).sum()), "new_errors":int((differences==-1).sum()),
             "prompt_cluster_bootstrap_95_percent_interval":np.quantile(bootstrap,[0.025,0.975]).tolist(),
             "validation_prompts":len(grouped), "validation_rows":len(valid),
             "promote_under_frozen_rule":gain>=0.01, "submission_authorized":False,
             "prediction_sha256":hashes,
             "limitations":"One repeatedly inspected pilot fold; bootstrap is diagnostic, not a new release gate. A promoted candidate requires broader prompt-held-out confirmation."}
    rows = []
    for index,label in enumerate(labels):
        mask = expected==index
        rows.append({"model":str(label),"rows":int(mask.sum()),
                     **{v:float(correct[v][mask].mean()) for v in correct}})
    pd.DataFrame(rows).to_csv(folder / "per-class-comparison.csv", index=False)
    (folder / "evaluation-audit.json").write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    evaluate(parser.parse_args().folder)
