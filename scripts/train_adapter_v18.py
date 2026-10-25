"""Standalone GPU worker for the frozen v18 paired-objective pilot."""
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def supcon_loss(features, labels, temperature=0.1):
    """Mean log-probability of same-class positives, excluding self pairs."""
    z = F.normalize(features.float(), dim=1)
    logits = z @ z.T / temperature
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    counts = positives.sum(1)
    if not bool((counts > 0).all()):
        raise ValueError("Every contrastive anchor must have a distinct positive.")
    denominator = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
    return -((logits - denominator[:, None]).masked_fill(~positives, 0).sum(1) / counts).mean()


def paired_batches(frame, steps, seed):
    """Two authors, two different prompts each; identical schedule for both arms."""
    rng = np.random.default_rng(seed)
    groups = {label: {p: sub.index.to_numpy() for p, sub in group.groupby("group")}
              for label, group in frame.groupby("label")}
    assert all(len(g) >= 2 for g in groups.values())
    labels = sorted(groups)
    for _ in range(steps):
        selected = []
        for label in rng.choice(labels, 2, replace=False):
            for prompt in rng.choice(list(groups[label]), 2, replace=False):
                selected.append(int(rng.choice(groups[label][prompt])))
        yield selected


class AttributionModel(nn.Module):
    def __init__(self, model_dir, plan):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16,
                                            local_files_only=True, attn_implementation="sdpa")
        backbone.config.use_cache = False
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.encoder = get_peft_model(backbone, LoraConfig(r=plan["lora_rank"],
            lora_alpha=plan["lora_alpha"], target_modules=plan["target_modules"], lora_dropout=0.0,
            bias="none"))
        width = backbone.config.hidden_size
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 18)
        self.projection = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 128))

    def forward(self, batch):
        hidden = self.encoder(**batch, use_cache=False).last_hidden_state.float()
        mask = batch["attention_mask"].clone()
        mask[:, 0] = 0
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        features = self.norm(pooled)
        return self.head(features), self.projection(features)


def main():
    import pandas as pd
    from transformers import AutoTokenizer
    import transformers, peft
    plan = json.loads(Path("plan.json").read_text())
    variant = os.environ["V18_VARIANT"]
    assert variant in plan["variants"]
    out = Path(variant)
    out.mkdir(exist_ok=True)
    started = time.monotonic()
    def log(**kw):
        record = dict(variant=variant, seconds=round(time.monotonic()-started, 2), **kw)
        print(json.dumps(record), flush=True)
        with (out / "progress.jsonl").open("a") as f:
            f.write(json.dumps(record)+"\n")
    assert torch.cuda.device_count() == 1
    torch.manual_seed(plan["seed"])
    np.random.seed(plan["seed"])
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    model_dir = plan["model_dir"]
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    frame = pd.read_parquet("training_frame.parquet")
    valid = pd.read_parquet("validation_frame.parquet")
    assert set(frame.loc[frame.stage == "host", "group"]).isdisjoint(set(valid.group))
    def tokenize(df):
        enc = tok(df.text.tolist(), truncation=True, max_length=plan["max_length"]-1,
                  add_special_tokens=False)
        return [ids+[tok.eos_token_id] for ids in enc["input_ids"]]
    tokens, val_tokens = tokenize(frame), tokenize(valid)
    def batch(ids, bank):
        return {k: v.cuda() for k,v in tok.pad({"input_ids": [bank[i] for i in ids]},
                                              padding=True, return_tensors="pt").items()}
    model = AttributionModel(model_dir, plan).cuda()
    trainable = [p for p in model.parameters() if p.requires_grad]
    # Frozen base is fp16; trainable adapters and heads must be fp32 for GradScaler.
    for p in trainable:
        p.data = p.data.float()
    log(event="loaded", trainable_parameters=sum(p.numel() for p in trainable),
        torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
        gpu=torch.cuda.get_device_name(0), training_rows=len(frame), validation_rows=len(valid))
    scaler = torch.amp.GradScaler("cuda")
    skipped = 0
    for stage, epochs in [("pool", plan["pool_epochs"]), ("host", plan["host_epochs"])]:
        subset = frame.loc[frame.stage == stage]
        micro_steps = math.ceil(len(subset)/plan["batch_size"])
        optimizer = torch.optim.AdamW([
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": plan[stage+"_adapter_lr"]},
            {"params": list(model.norm.parameters())+list(model.head.parameters())+list(model.projection.parameters()),
             "lr": plan[stage+"_head_lr"]}], weight_decay=0.01)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for step, ids in enumerate(paired_batches(subset, micro_steps, plan["seed"] + epoch + (100 if stage == "host" else 0))):
                if time.monotonic()-started > plan["wall_limit_seconds"]:
                    raise TimeoutError("Frozen pilot wall budget exhausted; no complete result claimed")
                labels = torch.as_tensor(frame.loc[ids, "label"].to_numpy(), device="cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    logits, features = model(batch(ids, tokens))
                    ce = F.cross_entropy(logits.float(), labels)
                    contrastive = supcon_loss(features, labels, plan["contrastive_temperature"])
                    loss = ce + (plan["contrastive_weight"] * contrastive if variant == "ce_supcon" else 0)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                window_start = (step // plan["gradient_accumulation"]) * plan["gradient_accumulation"]
                divisor = min(plan["gradient_accumulation"], micro_steps-window_start)
                scaler.scale(loss/divisor).backward()
                losses.append(float(loss.detach()))
                if (step+1) % plan["gradient_accumulation"] == 0 or step+1 == micro_steps:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    skipped += int(scaler.get_scale() < before)
                    optimizer.zero_grad(set_to_none=True)
                if step % 50 == 0:
                    log(event="train", stage=stage, epoch=epoch, step=step, steps=micro_steps,
                        mean_loss=float(np.mean(losses[-50:])), ce=float(ce.detach()),
                        contrastive=float(contrastive.detach()), skipped_steps=skipped,
                        peak_gb=torch.cuda.max_memory_allocated()/2**30)
            log(event="epoch_complete", stage=stage, epoch=epoch, mean_loss=float(np.mean(losses)))
        checkpoint = out / ("external_only" if stage == "pool" else "final")
        model.encoder.save_pretrained(checkpoint / "adapter")
        torch.save({"norm": model.norm.state_dict(), "head": model.head.state_dict(),
                    "projection": model.projection.state_dict()}, checkpoint / "heads.pt")
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(valid), 4):
            ids = list(range(start, min(start+4, len(valid))))
            with torch.autocast("cuda", dtype=torch.float16):
                logits, _ = model(batch(ids, val_tokens))
            scores.append(logits.float().cpu().numpy())
    scores = np.concatenate(scores)
    assert np.isfinite(scores).all()
    np.savez_compressed(out / "validation.npz", ids=valid.id.to_numpy(dtype=str),
                        labels=np.asarray(plan["labels"], dtype=str), logits=scores,
                        targets=valid.label.to_numpy())
    result = {"variant":variant, "accuracy":float((scores.argmax(1)==valid.label.to_numpy()).mean()),
              "seconds":time.monotonic()-started, "skipped_steps":skipped,
              "validation_rows":len(valid), "completed":True}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    log(event="complete", result=result)


if __name__ == "__main__":
    main()
