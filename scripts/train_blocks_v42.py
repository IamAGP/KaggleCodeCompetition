"""One isolated V42 fit. An external provider watchdog must bound billing."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from model_blocks_v40 import AttributionHeads, masked_mean, supervised_contrastive, balanced_accuracy
from sampling_v42 import coverage_batches


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arm', choices=['dense_ce', 'blocks_ce', 'dense_supcon', 'blocks_supcon'], required=True)
    parser.add_argument('--fold', type=int, required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    assert args.fold in plan['folds']
    root, out = Path(args.input_root), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def log(event, **values):
        record = dict(values)
        record.update(event=event, elapsed_seconds=time.monotonic()-start, arm=args.arm, fold=args.fold)
        with (out/'progress.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
    def deadline():
        if time.monotonic()-start > plan['per_fit_seconds']:
            raise TimeoutError('V42 fit deadline: partial checkpoints retained')
    for relative, digest in plan['inputs'].items():
        assert sha(root/relative) == digest, relative
    host = pd.read_csv(root/'train.csv', dtype={'id':str, 'prompt_id':str})
    folds = pd.read_csv(root/'folds.csv', dtype={'id':str, 'prompt_id':str})
    assert host.id.equals(folds.id) and host.prompt_id.equals(folds.prompt_id)
    assert folds.groupby('prompt_id').fold.nunique().max() == 1
    assert host.id.is_unique
    host['label'] = host.model.map({label:i for i,label in enumerate(plan['labels'])})
    assert host.label.notna().all()
    host['group'] = host.prompt_id
    host['stage'] = 'host'
    fit = host.loc[folds.fold != args.fold].copy()
    valid = host.loc[folds.fold == args.fold].copy()
    assert set(fit.group).isdisjoint(valid.group) and set(fit.id).isdisjoint(valid.id)
    (out/'split.json').write_text(json.dumps({'fit_ids':fit.id.tolist(), 'validation_ids':valid.id.tolist()}, indent=2))
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    assert torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
    torch.manual_seed(plan['seed'])
    np.random.seed(plan['seed'])
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.eos_token_id is not None
    encoded = tokenizer(host.text.tolist(), add_special_tokens=False, truncation=True, max_length=plan['max_length']-1)
    tokens = [ids + [tokenizer.eos_token_id] for ids in encoded['input_ids']]
    def batch(ids):
        return {k:v.cuda() for k,v in tokenizer.pad({'input_ids':[tokens[int(i)] for i in ids]}, return_tensors='pt').items()}
    backbone = AutoModel.from_pretrained(args.model_dir, local_files_only=True,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={'':0}, attn_implementation='sdpa', torch_dtype=torch.bfloat16)
    assert backbone.config.hidden_size == 2560 and backbone.config.num_hidden_layers == 36
    backbone.config.use_cache = False
    backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False})
    encoder = PeftModel.from_pretrained(backbone, root/'external/adapter', is_trainable=True)
    adapters = {name:p for name,p in encoder.named_parameters() if p.requires_grad}
    assert len(adapters) == 144 and all('lora_' in name for name in adapters)
    original = {name:p.detach().cpu().clone() for name,p in adapters.items()}
    torch.manual_seed(plan['seed'])
    heads = AttributionHeads(2560, args.arm.startswith('blocks')).cuda()
    external = torch.load(root/'external/heads.pt', map_location='cpu', weights_only=True)
    heads.norm.load_state_dict(external['norm'], strict=True)
    del external
    optimizer = torch.optim.AdamW([
        {'params':list(adapters.values()), 'lr':plan['adapter_lr']},
        {'params':list(heads.parameters()), 'lr':plan['head_lr']}], weight_decay=plan['weight_decay'])
    params = list(adapters.values()) + list(heads.parameters())
    log('loaded', gpu=torch.cuda.get_device_name(), fit_rows=len(fit), validation_rows=len(valid),
        adapter_parameters=sum(p.numel() for p in adapters.values()), head_parameters=sum(p.numel() for p in heads.parameters()))
    (out/'run-config.json').write_text(json.dumps(dict(plan=plan, arm=args.arm, fold=args.fold), indent=2))
    def checkpoint(name):
        folder = out/name
        encoder.save_pretrained(folder/'adapter')
        torch.save(heads.state_dict(), folder/'heads.pt')
    for epoch in range(plan['epochs']):
        schedule = coverage_batches(fit, plan['seed']+epoch)
        steps = len(schedule)
        covered = {i for ids in schedule for i in ids}
        assert covered == set(fit.index) and covered.isdisjoint(valid.index)
        log('coverage', epoch=epoch, rows=len(covered), batches=steps, presentations=4*steps)
        encoder.train(); heads.train(); optimizer.zero_grad(set_to_none=True)
        for step, ids in enumerate(schedule):
            deadline()
            y = torch.as_tensor(host.loc[ids, 'label'].to_numpy(), device='cuda')
            inputs = batch(ids)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hidden = encoder(**inputs, use_cache=False).last_hidden_state
                scores, projection, _ = heads(masked_mean(hidden, inputs['attention_mask']))
                ce = F.cross_entropy(scores.float(), y)
                contrastive = supervised_contrastive(projection, y, plan['temperature'])
                loss = ce + (plan['contrastive_weight']*contrastive if args.arm.endswith('supcon') else 0)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            window = step // plan['accumulation'] * plan['accumulation']
            (loss/min(plan['accumulation'], steps-window)).backward()
            if (step+1)%plan['accumulation'] == 0 or step+1 == steps:
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if step%25 == 0:
                log('train', epoch=epoch, step=step, steps=steps, ce=float(ce.detach()),
                    contrastive=float(contrastive.detach()), loss=float(loss.detach()),
                    peak_gib=torch.cuda.max_memory_allocated()/2**30)
        checkpoint(f'epoch-{epoch+1}')
        log('epoch_saved', epoch=epoch+1)
    encoder.eval(); heads.eval()
    scores, features = [], []
    with torch.inference_mode():
        for start_idx in range(0,len(valid),plan['batch_size']):
            deadline()
            inputs = batch(valid.index[start_idx:start_idx+plan['batch_size']])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hidden = encoder(**inputs, use_cache=False).last_hidden_state
                logits, _, z = heads(masked_mean(hidden, inputs['attention_mask']))
            scores.append(logits.float().cpu().numpy()); features.append(z.float().cpu().numpy())
            if start_idx%100 == 0: log('validation', rows_completed=start_idx+len(logits))
    scores = np.concatenate(scores); features = np.concatenate(features)
    assert np.isfinite(scores).all() and np.isfinite(features).all()
    targets = valid.label.to_numpy()
    np.savez_compressed(out/'validation.npz', ids=valid.id.to_numpy(dtype=str),
        prompts=valid.prompt_id.to_numpy(dtype=str), logits=scores, targets=targets,
        labels=np.asarray(plan['labels']), features=features)
    changed = sum(not torch.equal(original[name], p.detach().cpu()) for name,p in adapters.items())
    assert changed > 0
    result = dict(completed=True, arm=args.arm, fold=args.fold, rows=len(valid),
        balanced_accuracy=balanced_accuracy(targets,scores.argmax(1)),
        accuracy=float((targets == scores.argmax(1)).mean()), changed_adapter_tensors=changed,
        feature_std_mean=float(features.std(0).mean()), seconds=time.monotonic()-start)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    log('complete', **result)


if __name__ == '__main__':
    main()
