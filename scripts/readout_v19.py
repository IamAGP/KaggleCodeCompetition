"""Matched frozen readouts for base and v18 adapters; one fixed host fold."""
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel


def main():
    plan = json.loads(Path("plan.json").read_text())
    host = pd.read_csv(plan["host_path"], dtype={"id":str})
    upstream = Path(plan["upstream_path"])
    original_plan = json.loads((upstream / "plan.json").read_text())
    assert original_plan["pilot_fold"] == plan["pilot_fold"]
    fitting = pd.read_parquet(upstream / "training_frame.parquet")
    validation = pd.read_parquet(upstream / "validation_frame.parquet")
    fit_ids = set(fitting.loc[fitting.stage == "host", "id"])
    val_ids = set(validation.id)
    assert fit_ids.isdisjoint(val_ids) and fit_ids | val_ids == set(host.id)
    assert len(fit_ids) == 2159 and len(val_ids) == 540
    train_mask = host.id.isin(fit_ids).to_numpy()
    valid_mask = host.id.isin(val_ids).to_numpy()
    assert host.loc[valid_mask,"id"].tolist() == validation.id.tolist()
    labels = np.asarray(original_plan["labels"])
    assert np.array_equal(labels, np.sort(host.model.unique()))
    tok = AutoTokenizer.from_pretrained(plan["model_dir"], local_files_only=True)
    tok.padding_side = "right"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tokenized = tok(host.text.tolist(), truncation=True, max_length=plan["max_length"]-1,
                    add_special_tokens=False)["input_ids"]
    tokenized = [ids+[tok.eos_token_id] for ids in tokenized]
    order = np.argsort([len(ids) for ids in tokenized], kind="stable")
    checkpoint_paths = {"ce_external":"ce/external_only/adapter", "ce_final":"ce/final/adapter",
                        "supcon_external":"ce_supcon/external_only/adapter", "supcon_final":"ce_supcon/final/adapter"}
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    assert torch.cuda.device_count() == 1
    for variant in json.loads(os.environ["V19_VARIANTS"]):
        started = time.monotonic()
        print(json.dumps({"variant":variant,"event":"loading"}), flush=True)
        model = AutoModel.from_pretrained(plan["model_dir"], torch_dtype=torch.float16,
                                         attn_implementation="sdpa", local_files_only=True)
        if variant != "base":
            checkpoint = upstream / checkpoint_paths[variant]
            weight = checkpoint / "adapter_model.safetensors"
            assert hashlib.sha256(weight.read_bytes()).hexdigest() == plan["adapter_sha256"][variant]
            model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
        model = model.cuda().eval()
        width = model.config.hidden_size
        features = np.empty((len(host),len(plan["layers"]),width), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0,len(host),plan["batch_size"]):
                ids = order[start:start+plan["batch_size"]]
                enc = {k:v.cuda() for k,v in tok.pad({"input_ids":[tokenized[i] for i in ids]},
                                                    padding=True,return_tensors="pt").items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    states = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
                mask = enc["attention_mask"].clone(); mask[:,0] = 0
                for j,layer in enumerate(plan["layers"]):
                    h = states[layer].float()
                    features[ids,j] = ((h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)).cpu().numpy()
                del states
                if start % 400 == 0:
                    print(json.dumps({"variant":variant,"completed":min(start+len(ids),len(host)),
                                      "total":len(host),"seconds":time.monotonic()-started}), flush=True)
        assert np.isfinite(features).all()
        np.save(variant+"_features.npy", features)
        del model
        gc.collect(); torch.cuda.empty_cache()
        predictions = []
        diagnostics = {}
        for j,layer in enumerate(plan["layers"]):
            scaler = StandardScaler()
            X = scaler.fit_transform(features[train_mask,j])
            V = scaler.transform(features[valid_mask,j])
            clf = LogisticRegression(C=plan["C"], max_iter=plan["max_iter"], random_state=plan["seed"])
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                clf.fit(X,host.loc[train_mask,"model"])
            assert np.array_equal(clf.classes_, labels)
            scores = clf.predict_log_proba(V)
            predictions.append(scores)
            diagnostics[str(layer)] = float((labels[scores.argmax(1)]==host.loc[valid_mask,"model"]).mean())
        scores = np.mean(predictions,axis=0)
        accuracy = float((labels[scores.argmax(1)]==host.loc[valid_mask,"model"]).mean())
        np.savez_compressed(variant+"_validation.npz", ids=host.loc[valid_mask,"id"].to_numpy(dtype=str),
                            labels=labels.astype(str), scores=scores, per_layer_scores=np.stack(predictions))
        result = {"variant":variant,"accuracy":accuracy,"per_layer_accuracy":diagnostics,
                  "seconds":time.monotonic()-started,"fit_rows":int(train_mask.sum()),
                  "validation_rows":int(valid_mask.sum()),"feature_shape":list(features.shape),
                  "feature_sha256":hashlib.sha256(Path(variant+"_features.npy").read_bytes()).hexdigest()}
        Path(variant+"_result.json").write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        del features
        gc.collect()


if __name__ == "__main__":
    main()
