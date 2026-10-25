"""Verify v19 feature provenance and fixed pilot comparisons."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    root = Path("artifacts/improvement-v19/kaggle-output")
    frozen = json.loads(Path("artifacts/improvement-v19/frozen-plan.json").read_text())
    summary = json.loads((root/"v19_summary.json").read_text())
    plan = json.loads((root/"plan.json").read_text())
    assert hashlib.sha256((root/"worker.py").read_bytes()).hexdigest() == frozen["worker_sha256"]
    for key,value in frozen.items():
        if key not in {"notebook_sha256","worker_sha256"}:
            assert plan[key] == value, key
    host_path = Path("data/name-that-llm/train.csv")
    assert hashlib.sha256(host_path.read_bytes()).hexdigest() == plan["host_sha256"]
    host = pd.read_csv(host_path,dtype={"id":str,"prompt_id":str})
    folds = pd.read_csv("artifacts/baseline-v1/folds.csv",dtype={"id":str,"prompt_id":str})
    assert folds.id.equals(host.id) and folds.prompt_id.equals(host.prompt_id)
    valid = host.loc[folds.fold==plan["pilot_fold"]].reset_index(drop=True)
    labels = np.sort(host.model.unique())
    correct = {}
    results = {}
    for variant in plan["variants"]:
        path = root/(variant+"_features.npy")
        feature_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        reported = summary["results"][variant]
        assert feature_hash == reported["feature_sha256"]
        features = np.load(path,mmap_mode="r")
        assert features.shape == (len(host),3,2560) and np.isfinite(features).all()
        with np.load(root/(variant+"_validation.npz"),allow_pickle=False) as data:
            assert np.array_equal(data["ids"],valid.id.to_numpy())
            assert np.array_equal(data["labels"],labels)
            scores = data["scores"]
            assert scores.shape == (len(valid),18) and np.isfinite(scores).all()
            assert np.allclose(scores,data["per_layer_scores"].mean(0))
            correct[variant] = labels[scores.argmax(1)]==valid.model.to_numpy()
        accuracy = float(correct[variant].mean())
        assert abs(accuracy-reported["accuracy"])<1e-12
        results[variant] = {"accuracy":accuracy,"feature_sha256":feature_hash}
    comparisons = {}
    for reference in ["base","ce_final"]:
        delta = correct["supcon_final"].astype(float)-correct[reference].astype(float)
        groups = pd.DataFrame({"prompt":valid.prompt_id,"delta":delta}).groupby("prompt").delta.agg(["sum","size"])
        draws = np.random.default_rng(1901).integers(len(groups),size=(10000,len(groups)))
        boot = groups["sum"].to_numpy()[draws].sum(1)/groups["size"].to_numpy()[draws].sum(1)
        comparisons[reference] = {"gain":float(delta.mean()),"corrected_errors":int((delta==1).sum()),
                                 "new_errors":int((delta==-1).sum()),"prompt_bootstrap_95_interval":np.quantile(boot,[0.025,0.975]).tolist()}
    promoted = [v for v,r in results.items() if v!="base" and r["accuracy"]-results["base"]["accuracy"]>=0.01 and r["accuracy"]>=0.4044444444]
    assert promoted == summary["promoted_for_confirmation"]
    outcome = {"status":"complete","results":results,"supcon_final_comparisons":comparisons,
               "promoted_for_confirmation":promoted,"submitted":False,
               "limitations":"One repeatedly inspected 540-row/30-prompt fold; no leaderboard inference from this pilot."}
    (root/"evaluation-audit.json").write_text(json.dumps(outcome,indent=2))
    print(json.dumps(outcome,indent=2))


if __name__=="__main__":main()
