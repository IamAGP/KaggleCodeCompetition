"""Deterministic training-frame preparation and conservative external overlap screen."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def prepare(host_path, pool_path, fold_by_prompt, plan, output):
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    host = pd.read_csv(host_path, dtype={"id":str, "prompt_id":str})
    pool = pd.read_parquet(pool_path)
    assert hashlib.sha256(Path(host_path).read_bytes()).hexdigest() == plan["host_sha256"]
    assert hashlib.sha256(Path(pool_path).read_bytes()).hexdigest() == plan["pool_sha256"]
    assert set(host.prompt_id) == set(fold_by_prompt)
    labels = sorted(host.model.unique())
    assert len(labels) == 18 and set(pool.model) <= set(labels)
    def normalize(s):
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s))).strip().lower()
    hashes = pool.text.map(normalize)
    ambiguous = pool.assign(key=hashes).groupby("key").model.nunique()
    reject = hashes.isin(set(host.text.map(normalize))) | hashes.isin(ambiguous[ambiguous>1].index) | hashes.duplicated()
    print(json.dumps({"event":"screen_start", "pool_rows":len(pool)}), flush=True)
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True, dtype=np.float32)
    X = vectorizer.fit_transform(pd.concat([host.text, pool.text], ignore_index=True))
    H, P = X[:len(host)], X[len(host):]
    similarity = np.zeros(len(pool), dtype=np.float32)
    for start in range(0, len(pool), 512):
        similarity[start:start+512] = (P[start:start+512] @ H.T).max(axis=1).toarray().ravel()
    reject |= similarity >= plan["external_host_cosine_exclusion"]
    clean = pool.loc[~reject]
    selected = pd.concat([g.sample(min(len(g), plan["external_cap_per_class"]), random_state=plan["seed"])
                          for _,g in clean.groupby("model")], ignore_index=True)
    selected["group"] = "external:" + selected.source.astype(str) + ":" + selected.prompt_key.astype(str)
    selected["stage"] = "pool"
    host["group"] = "host:" + host.prompt_id
    host["stage"] = "host"
    valid_mask = host.prompt_id.map(fold_by_prompt) == plan["pilot_fold"]
    train = pd.concat([selected, host.loc[~valid_mask]], ignore_index=True)
    valid = host.loc[valid_mask].reset_index(drop=True)
    for frame in [train, valid]:
        frame["label"] = frame.model.map({s:i for i,s in enumerate(labels)}).astype(int)
    train.to_parquet(output / "training_frame.parquet", index=False)
    valid.to_parquet(output / "validation_frame.parquet", index=False)
    audit = {"pool_rows":len(pool), "excluded_rows":int(reject.sum()),
             "near_host_rows":int((similarity >= plan["external_host_cosine_exclusion"]).sum()),
             "selected_pool_rows":len(selected), "host_fit_rows":int((~valid_mask).sum()),
             "host_validation_rows":int(valid_mask.sum()), "labels":labels,
             "training_frame_sha256":hashlib.sha256((output / "training_frame.parquet").read_bytes()).hexdigest(),
             "validation_frame_sha256":hashlib.sha256((output / "validation_frame.parquet").read_bytes()).hexdigest()}
    (output / "data-audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)
    return labels
