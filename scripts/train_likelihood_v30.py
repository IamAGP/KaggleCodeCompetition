"""Fixed two-stage, two-GPU generative attribution pilot; no model selection."""
import hashlib
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
import time

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from safetensors.torch import save_file,load_file
from transformers import AutoModelForCausalLM,AutoTokenizer
from source_likelihood import SourceLanguageModel
from likelihood_batches import epoch_windows


def main():
    started=time.monotonic()
    plan=json.loads(Path('plan.json').read_text())
    assert plan['feasibility_audit']['verified'] and plan['feasibility_audit']['qualifies_for_pilot']
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    world=dist.get_world_size();assert world==plan['world_size']==2
    def log(**values):
        values.update(rank=rank,elapsed_seconds=time.monotonic()-started)
        with Path(f'progress-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(values)+'\n')
        print(json.dumps(values),flush=True)
    torch.manual_seed(plan['training_seed'])
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    frame=pd.read_parquet('training_frame.parquet')
    valid=pd.read_parquet('validation_frame.parquet')
    assert len(frame)==20159 and len(valid)==540
    assert set(frame.loc[frame.stage=='host','group']).isdisjoint(valid.group)
    labels=np.array(plan['labels']);assert len(labels)==18
    tok=AutoTokenizer.from_pretrained(plan['model_dir'],local_files_only=True)
    tok.padding_side='right'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    def tokenize(data):
        tokens=tok(data.text.tolist(),truncation=True,max_length=plan['max_length']-2,add_special_tokens=False)['input_ids']
        return [[tok.eos_token_id]+ids+[tok.eos_token_id] for ids in tokens]
    train_tokens=tokenize(frame);valid_tokens=tokenize(valid)
    def batch(tokens,indices):
        return tok.pad({'input_ids':[tokens[i] for i in indices]},padding=True,return_tensors='pt').to('cuda')
    backbone=AutoModelForCausalLM.from_pretrained(plan['model_dir'],torch_dtype=torch.float16,
                attn_implementation='sdpa',local_files_only=True).cuda()
    model=SourceLanguageModel(backbone,18,plan['rank'],plan['alpha'],plan['dropout'],plan['head_chunk'])
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train();parallel=DistributedDataParallel(model,device_ids=[rank])
    parameters=[p for p in model.parameters() if p.requires_grad]
    if rank==0:save_file(model.adapter_state(),'initial-adapter.safetensors')
    scaler=torch.amp.GradScaler('cuda',init_scale=plan['initial_loss_scale'],growth_interval=plan['loss_scale_growth_interval'])
    targets=torch.tensor(frame.label.to_numpy(),device='cuda')
    total_skipped=0;total_updates=0
    log(event='loaded',trainable_parameters=sum(p.numel() for p in parameters),
        training_rows=len(frame),validation_rows=len(valid),device=torch.cuda.get_device_name())
    for stage in ['pool','host']:
        stage_rows=np.flatnonzero(frame.stage.to_numpy()==stage)
        epochs=plan[stage+'_epochs'];lr=plan[stage+'_lr']
        updates_per_epoch=math.ceil(len(stage_rows)/(world*plan['batch_size']*plan['gradient_accumulation']))
        stage_updates=epochs*updates_per_epoch;warmup=max(1,math.ceil(stage_updates*plan['warmup_fraction']))
        optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=plan['weight_decay'])
        stage_step=0
        for epoch in range(epochs):
            loss_sum=0.;examples=0
            windows=epoch_windows(len(stage_rows),plan['training_seed'],epoch+(100 if stage=='host' else 0),
                                  rank,world,plan['batch_size'],plan['gradient_accumulation'])
            for window_index,steps in enumerate(windows):
                if time.monotonic()-started>plan['wall_limit_seconds']:raise RuntimeError('Training wall limit exceeded')
                stage_step+=1
                factor=stage_step/warmup if stage_step<=warmup else .5*(1+math.cos(math.pi*(stage_step-warmup)/max(stage_updates-warmup,1)))
                for group in optimizer.param_groups:group['lr']=lr*factor
                optimizer.zero_grad(set_to_none=True)
                for local,weight,synchronize in steps:
                    selected=stage_rows[local] if len(local) else stage_rows[:1]
                    data=batch(train_tokens,selected)
                    with (nullcontext() if synchronize else parallel.no_sync()):
                        with torch.autocast('cuda',dtype=torch.float16):
                            raw=parallel(data['input_ids'],data['attention_mask'],targets[selected])
                            loss=raw*weight
                        assert torch.isfinite(raw)
                        scaler.scale(loss).backward()
                    loss_sum+=float(raw.detach())*len(local);examples+=len(local)
                scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(parameters,plan['gradient_clip'])
                old=scaler.get_scale();scaler.step(optimizer);scaler.update()
                skipped=scaler.get_scale()<old;total_skipped+=int(skipped);total_updates+=1
                if skipped or window_index%50==0:
                    log(event='update',stage=stage,epoch=epoch,window=window_index,gradient_norm=float(norm),
                        skipped=skipped,scale=scaler.get_scale(),lr=lr*factor)
                assert total_skipped<=max(2,int(.01*total_updates)), 'Excess numerical skips'
            sums=torch.tensor([loss_sum,examples],device='cuda',dtype=torch.float64)
            dist.all_reduce(sums)
            assert int(sums[1])==len(stage_rows)
            log(event='epoch_complete',stage=stage,epoch=epoch,mean_loss=float(sums[0]/sums[1]),rows=int(sums[1]))
        name='external-only' if stage=='pool' else 'final'
        if rank==0:
            state=model.adapter_state();assert all(torch.isfinite(v).all() for v in state.values())
            save_file(state,name+'-adapter.safetensors')
        dist.barrier()
    # Evaluation starts only after the fixed final epoch; no best-epoch selection.
    model.eval()
    selected=np.arange(rank,len(valid),world)
    clipped=[];means=[]
    for start in range(0,len(selected),plan['batch_size']):
        if time.monotonic()-started>plan['wall_limit_seconds']:raise RuntimeError('Evaluation wall limit exceeded')
        indices=selected[start:start+plan['batch_size']]
        data=batch(valid_tokens,indices)
        with torch.autocast('cuda',dtype=torch.float16):
            result=model.all_source_scores(data['input_ids'],data['attention_mask'])
        clipped.append(result['clipped'].cpu().numpy());means.append(result['mean'].cpu().numpy())
        if start==0:
            model.load_adapter_state(load_file('final-adapter.safetensors'))
            with torch.autocast('cuda',dtype=torch.float16):reloaded=model.all_source_scores(data['input_ids'],data['attention_mask'])
            torch.testing.assert_close(reloaded['clipped'],result['clipped'],atol=1e-5,rtol=1e-4)
        if start%40==0:log(event='validation',completed=min(start+plan['batch_size'],len(selected)),total=len(selected))
    scores=np.concatenate(clipped);mean=np.concatenate(means)
    assert np.isfinite(scores).all() and np.isfinite(mean).all()
    np.savez_compressed(f'validation-rank{rank}.npz',ids=valid.iloc[selected].id.astype(str).to_numpy(),
                        labels=labels,clipped=scores,mean=mean)
    log(event='complete',skipped_steps=total_skipped,optimizer_steps=total_updates,
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
