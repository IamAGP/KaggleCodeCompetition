"""Bounded 8B NF4 feasibility probe; does not estimate competition accuracy."""
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file
from peft import set_peft_model_state_dict


def main():
    started = time.monotonic()
    plan = json.loads(Path('plan.json').read_text())
    def log(**values):
        record = dict(seconds=time.monotonic()-started, **values)
        with Path('progress.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
    torch.manual_seed(plan['seed'])
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    configs = []
    for path in Path('/kaggle/input').rglob('config.json'):
        cfg = json.loads(path.read_text())
        log(event='input_config', path=str(path), model_type=cfg.get('model_type'),
            hidden_size=cfg.get('hidden_size'), num_hidden_layers=cfg.get('num_hidden_layers'))
        if cfg.get('model_type') == 'qwen3' and cfg.get('hidden_size') == 4096 and cfg.get('num_hidden_layers') == 36:
            configs.append(path.parent)
    assert len(configs) == 1, configs
    model_dir = configs[0]
    train_files = list(Path('/kaggle/input').rglob('train.csv'))
    assert len(train_files) == 1
    assert hashlib.sha256(train_files[0].read_bytes()).hexdigest() == plan['train_sha256']
    train = pd.read_csv(train_files[0])
    labels = sorted(train.model.unique())
    # Long real passages exercise the full padded sequence length, with no validation selection.
    frame = train.assign(length=train.text.str.len()).sort_values(['length', 'id'], ascending=[False, True]).head(32)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tok.padding_side = 'right'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tokens = tok(frame.text.tolist(), add_special_tokens=False, truncation=True, max_length=255)['input_ids']
    tokens = [ids+[tok.eos_token_id] for ids in tokens]
    bank = tok.pad({'input_ids': tokens}, padding='max_length', max_length=256, return_tensors='pt')
    targets = torch.tensor([labels.index(label) for label in frame.model], device='cuda:0')
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    backbone = AutoModel.from_pretrained(model_dir, quantization_config=quant, torch_dtype=torch.float16,
        device_map={'': 0}, local_files_only=True, attn_implementation='sdpa')
    backbone.config.use_cache = False
    backbone = prepare_model_for_kbit_training(backbone,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    encoder = get_peft_model(backbone, LoraConfig(r=16, lora_alpha=32,
        target_modules='all-linear', lora_dropout=0.0, bias='none'))
    head = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 18)).cuda()
    trainable = [p for p in encoder.parameters() if p.requires_grad]+list(head.parameters())
    for parameter in trainable:
        parameter.data = parameter.data.float()
    assert all('lora_' in name for name,p in encoder.named_parameters() if p.requires_grad)
    initial = {name: p.detach().cpu().clone() for name,p in encoder.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    log(event='loaded', versions={name:importlib.metadata.version(name) for name in
        ['torch','transformers','peft','bitsandbytes','accelerate']}, gpu=torch.cuda.get_device_name(0),
        trainable_parameters=sum(p.numel() for p in trainable), model_config_sha256=hashlib.sha256((model_dir/'config.json').read_bytes()).hexdigest())
    def forward(batch):
        hidden = encoder(**batch, use_cache=False).last_hidden_state.float()
        mask = batch['attention_mask'].clone()
        mask[:, 0] = 0
        pooled = (hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1, keepdim=True).clamp_min(1)
        return head(pooled)
    elapsed = []
    skipped = 0
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    for step in range(plan['microsteps']):
        assert time.monotonic()-started < plan['wall_limit_seconds']
        ids = np.arange((step%8)*4, (step%8+1)*4)
        batch = {key:value[ids].cuda() for key,value in bank.items()}
        torch.cuda.synchronize()
        tick = time.monotonic()
        with torch.autocast('cuda', dtype=torch.float16):
            logits = forward(batch)
            loss = nn.functional.cross_entropy(logits.float(), targets[ids])
        assert torch.isfinite(loss), 'nonfinite loss'
        scaler.scale(loss/8).backward()
        if (step+1)%8 == 0:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < before)
            log(event='update', step=step, grad_norm=float(norm), skipped=skipped)
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed.append(time.monotonic()-tick)
        log(event='step', step=step, loss=float(loss.detach()), step_seconds=elapsed[-1],
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
    changed = sum(not torch.equal(initial[name],p.detach().cpu()) for name,p in encoder.named_parameters() if p.requires_grad)
    assert changed > 0 and skipped < plan['microsteps']//8
    assert all(torch.isfinite(p).all() for p in trainable)
    encoder.save_pretrained('adapter', safe_serialization=True)
    head.eval(); encoder.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        before = forward(batch).detach().float().cpu()
    # Perturb and restore adapters to exercise actual saved tensor loading.
    for p in encoder.parameters():
        if p.requires_grad:
            p.data.zero_()
    state = load_file('adapter/adapter_model.safetensors')
    loaded = set_peft_model_state_dict(encoder, state)
    assert not loaded.unexpected_keys
    assert not [key for key in loaded.missing_keys if 'lora_' in key]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        after = forward(batch).detach().float().cpu()
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-5)
    measured = elapsed[8:]
    result = dict(completed=True, kind='feasibility only, no accuracy result', microsteps=len(elapsed),
        changed_tensors=changed, skipped_updates=skipped, reload_check='passed',
        median_microstep_seconds=float(np.median(measured)), max_microstep_seconds=max(measured),
        projected_5580_microsteps_seconds=float(np.mean(measured)*5580),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        elapsed_seconds=time.monotonic()-started,
        caveats=['CE only, not SupCon; fixed length 256; throughput projection excludes loading and readout',
                 'No validation or public score; all-linear adapters differ from earlier q/v-only experiments'])
    Path('result.json').write_text(json.dumps(result, indent=2))
    log(event='complete', **result)


if __name__ == '__main__':
    main()
