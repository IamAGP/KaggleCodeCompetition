"""Bounded feasibility measurement; discard this checkpoint after the probe."""
import hashlib
import json
import os
from pathlib import Path
import time
from contextlib import nullcontext
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from source_likelihood import SourceLanguageModel


def main():
    started=time.monotonic()
    plan=json.loads(Path('plan.json').read_text())
    dist.init_process_group('nccl')
    rank=dist.get_rank()
    assert dist.get_world_size()==plan['world_size']==2
    torch.cuda.set_device(rank)
    def log(**record):
        record['elapsed_seconds']=time.monotonic()-started
        with Path(f'progress-rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    torch.manual_seed(plan['seed'])
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    candidates=[]
    for path in Path('/kaggle/input').rglob('config.json'):
        cfg=json.loads(path.read_text())
        if cfg.get('model_type')=='qwen2' and cfg.get('hidden_size')==1536 and cfg.get('num_hidden_layers')==28:
            candidates.append(path.parent)
    assert len(candidates)==1,candidates
    model_dir=candidates[0]
    files=list(Path('/kaggle/input').rglob('train.csv'))
    assert len(files)==1 and hashlib.sha256(files[0].read_bytes()).hexdigest()==plan['train_sha256']
    data=pd.read_csv(files[0])
    labels=sorted(data.model.unique())
    frame=data.assign(length=data.text.str.len()).sort_values(['length','id'],ascending=[False,True]).head(32)
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    tok.padding_side='right'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    texts=tok(frame.text.tolist(),add_special_tokens=False,truncation=True,max_length=plan['max_length']-2)['input_ids']
    texts=[[tok.eos_token_id]+ids+[tok.eos_token_id] for ids in texts]
    bank=tok.pad({'input_ids':texts},padding='max_length',max_length=plan['max_length'],return_tensors='pt')
    source=torch.tensor([labels.index(label) for label in frame.model],device='cuda')
    lm=AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype=torch.float16,local_files_only=True,
                                          attn_implementation='sdpa').cuda()
    model=SourceLanguageModel(lm,plan['sources'],plan['rank'],plan['alpha'],plan['dropout'],plan['head_chunk'])
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    parallel=DistributedDataParallel(model,device_ids=[rank])
    params=[p for p in model.parameters() if p.requires_grad]
    before=model.adapter_state()
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    opt.zero_grad(set_to_none=True)
    steps=[]
    log(event='loaded',trainable_parameters=sum(p.numel() for p in params),torch=torch.__version__,device=torch.cuda.get_device_name())
    for step in range(plan['microsteps']):
        begin=time.monotonic()
        offset=(step*plan['world_size']+rank)*plan['batch_size']
        rows=torch.arange(offset,offset+plan['batch_size'])%len(frame)
        ids=bank['input_ids'][rows].cuda();mask=bank['attention_mask'][rows].cuda()
        update=(step+1)%plan['gradient_accumulation']==0
        with (nullcontext() if update else parallel.no_sync()):
            with torch.autocast('cuda',dtype=torch.float16):
                loss=parallel(ids,mask,source[rows.cuda()])/plan['gradient_accumulation']
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
        if update:
            scaler.unscale_(opt)
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            assert torch.isfinite(norm)
            old=scaler.get_scale();scaler.step(opt);scaler.update()
            assert scaler.get_scale()>=old,'Skipped optimizer step'
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize();seconds=time.monotonic()-begin;steps.append(seconds)
        log(event='microstep',step=step,loss=float(loss.detach())*plan['gradient_accumulation'],seconds=seconds,update=update)
    state=model.adapter_state()
    assert all(torch.isfinite(v).all() for v in state.values())
    changed=sum(not torch.equal(before[k],v) for k,v in state.items())
    assert changed==len(state),(changed,len(state))
    save_file(state,f'probe-adapter-rank{rank}.safetensors')
    model.eval()
    begin=time.monotonic()
    with torch.autocast('cuda',dtype=torch.float16):
        scores=model.all_source_scores(ids,mask)['clipped']
    torch.cuda.synchronize();score_seconds=time.monotonic()-begin
    assert torch.isfinite(scores).all()
    with torch.no_grad():model.prefix.zero_()
    model.load_adapter_state(load_file(f'probe-adapter-rank{rank}.safetensors'))
    with torch.autocast('cuda',dtype=torch.float16):
        reloaded=model.all_source_scores(ids,mask)['clipped']
    torch.testing.assert_close(scores,reloaded,atol=1e-5,rtol=1e-4)
    per_step=float(np.mean(steps[8:]))
    pilot=per_step*plan['projected_training_microsteps']+score_seconds*plan['pilot_validation_rows']/plan['batch_size']+600
    hidden=score_seconds*plan['hidden_projection_rows']/plan['batch_size']+plan['other_ensemble_allowance_seconds']
    result={'verified_updates':True,'checkpoint_reload_verified':True,'changed_tensors':changed,
            'mean_training_microstep_seconds':per_step,'all_source_batch_seconds':score_seconds,
            'projected_pilot_seconds':pilot,'projected_hidden_seconds':hidden,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'qualifies_for_pilot':pilot<=plan['launch_gate_seconds'] and hidden<=plan['launch_gate_seconds'],
            'checkpoint_discard_required':True,'submission_authorized':False}
    Path(f'probe-result-rank{rank}.json').write_text(json.dumps(result,indent=2))
    log(event='complete',rank=rank,result=result)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':main()
