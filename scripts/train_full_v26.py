"""Prepared 8B full-host worker; launch requires a qualified V25 result.

Reuse the V21 resume/sampler/optimizer flow with the V25 NF4 encoder.
The delivery plan must set contrastive_weight=0 if the CE arm is selected.
"""
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoTokenizer
from v18_worker import AttributionModel, paired_batches, supcon_loss


def main():
    plan = json.loads(Path('plan.json').read_text())
    assert plan['experiment'] == 'improvement-v26'
    assert plan['pilot_qualification']['verified']
    selected = plan['pilot_qualification']['selected_for_full_host_training']
    assert selected in ('ce_final', 'supcon_final')
    params = plan['training_parameters']
    assert params['contrastive_weight'] == (0.0 if selected == 'ce_final' else 0.1)
    out = Path('full'); out.mkdir(exist_ok=True)
    started = time.monotonic()
    def log(**record):
        record['seconds'] = round(time.monotonic()-started, 2)
        print(json.dumps(record), flush=True)
        with (out/'progress.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    torch.manual_seed(params['seed']); np.random.seed(params['seed'])
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    assert torch.cuda.device_count() == 1
    frame = pd.read_csv(plan['host_path'], dtype={'id':str})
    assert len(frame)==2699 and frame.id.is_unique and frame.prompt_id.nunique()==150
    assert hashlib.sha256(Path(plan['host_path']).read_bytes()).hexdigest()==plan['host_sha256']
    labels = np.asarray(plan['labels'])
    assert np.array_equal(labels, np.sort(frame.model.unique()))
    frame['label'] = frame.model.map({v:i for i,v in enumerate(labels)})
    frame['group'] = frame.prompt_id
    assert frame.label.notna().all()
    tokenizer = AutoTokenizer.from_pretrained(plan['model_dir'], local_files_only=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(frame.text.tolist(), truncation=True, max_length=params['max_length']-1,
                       add_special_tokens=False)['input_ids']
    tokens = [ids+[tokenizer.eos_token_id] for ids in tokens]
    model = AttributionModel(plan['model_dir'], params)
    assert getattr(model.encoder, 'is_loaded_in_4bit', False)
    checkpoint = Path(plan['checkpoint'])
    for relative, digest in plan['checkpoint_sha256'].items():
        assert hashlib.sha256((checkpoint/relative).read_bytes()).hexdigest()==digest
    state = load_file(str(checkpoint/'adapter/adapter_model.safetensors'))
    result = set_peft_model_state_dict(model.encoder, state)
    assert not result.unexpected_keys, result.unexpected_keys
    assert not [k for k in result.missing_keys if 'lora_' in k], result.missing_keys
    heads = torch.load(checkpoint/'heads.pt', map_location='cpu', weights_only=True)
    for name in ['norm','head','projection']: getattr(model,name).load_state_dict(heads[name], strict=True)
    model.norm.cuda(); model.head.cuda(); model.projection.cuda()
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable: p.data = p.data.float()
    initial_adapter = {n:p.detach().cpu().clone() for n,p in model.encoder.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([
        {'params':[p for p in model.encoder.parameters() if p.requires_grad], 'lr':params['host_adapter_lr']},
        {'params':list(model.norm.parameters())+list(model.head.parameters())+list(model.projection.parameters()),
         'lr':params['host_head_lr']}], weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024); skipped = 0
    steps = math.ceil(len(frame)/params['batch_size'])
    log(event='loaded', rows=len(frame), prompts=frame.prompt_id.nunique(), steps_per_epoch=steps,
        trainable_parameters=sum(p.numel() for p in trainable))
    for epoch in range(params['host_epochs']):
        model.train(); optimizer.zero_grad(set_to_none=True); losses=[]
        for step,ids in enumerate(paired_batches(frame, steps, params['seed']+100+epoch)):
            if time.monotonic()-started>params['wall_limit_seconds']: raise TimeoutError('Training budget exhausted')
            batch = {k:v.cuda() for k,v in tokenizer.pad({'input_ids':[tokens[i] for i in ids]},
                                                       padding=True, return_tensors='pt').items()}
            targets = torch.as_tensor(frame.loc[ids,'label'].to_numpy(), device='cuda')
            with torch.autocast('cuda', dtype=torch.float16):
                logits, features = model(batch)
                ce = F.cross_entropy(logits.float(), targets)
                contrastive = supcon_loss(features, targets, params['contrastive_temperature'])
                loss = ce+params['contrastive_weight']*contrastive
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            window_start=(step//params['gradient_accumulation'])*params['gradient_accumulation']
            divisor=min(params['gradient_accumulation'],steps-window_start)
            scaler.scale(loss/divisor).backward(); losses.append(float(loss.detach()))
            if (step+1)%params['gradient_accumulation']==0 or step+1==steps:
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(trainable,1.0)
                before=scaler.get_scale(); scaler.step(optimizer); scaler.update()
                skipped+=int(scaler.get_scale()<before); optimizer.zero_grad(set_to_none=True)
            if step%50==0:
                log(event='train',epoch=epoch,step=step,steps=steps,mean_loss=float(np.mean(losses[-50:])),
                    ce=float(ce.detach()),contrastive=float(contrastive.detach()),skipped_steps=skipped)
        log(event='epoch_complete',epoch=epoch,mean_loss=float(np.mean(losses)))
    assert all(torch.isfinite(p).all() for p in trainable)
    changed=sum(not torch.equal(initial_adapter[n],p.detach().cpu()) for n,p in model.encoder.named_parameters() if p.requires_grad)
    assert changed>0
    model.encoder.save_pretrained(out/'adapter')
    torch.save({n:getattr(model,n).state_dict() for n in ['norm','head','projection']}, out/'heads.pt')
    result={'completed':True,'rows':len(frame),'prompts':frame.prompt_id.nunique(),
            'epochs':params['host_epochs'],'skipped_steps':skipped,'changed_adapter_tensors':changed,
            'seconds':time.monotonic()-started,'labels':labels.tolist()}
    (out/'training_result.json').write_text(json.dumps(result,indent=2))
    log(event='complete',result=result)


if __name__=='__main__': main()
