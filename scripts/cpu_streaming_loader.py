"""Tensor-at-a-time safetensors loader, scoped to retained Qwen3 backbone layers."""
import gc,json,time,os
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from transformers import AutoConfig,AutoModel


def memory_snapshot():
    def read(path):
        p=Path(path)
        return p.read_text().strip() if p.is_file() else None
    rss=int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))*1024
    return {'rss_bytes':rss,'cgroup_current':read('/sys/fs/cgroup/memory.current'),'cgroup_peak':read('/sys/fs/cgroup/memory.peak'),'cgroup_limit':read('/sys/fs/cgroup/memory.max')}


def load_streamed_trunk(base_dir,keep,dtype):
    base=Path(base_dir);config=AutoConfig.from_pretrained(base,local_files_only=True)
    assert config.model_type=='qwen3', 'This loader is explicitly scoped to Qwen3'
    config.num_hidden_layers=keep;config.use_cache=False
    if getattr(config,'layer_types',None) is not None:config.layer_types=config.layer_types[:keep]
    with init_empty_weights(include_buffers=False):
        model=AutoModel.from_config(config,dtype=dtype)
    model.norm=nn.Identity()
    expected={name:(tuple(p.shape),p.requires_grad) for name,p in model.named_parameters()}
    index=base/'model.safetensors.index.json'
    if index.exists():weights=json.loads(index.read_text())['weight_map']
    else:
        file=base/'model.safetensors'
        with safe_open(file,framework='pt',device='cpu') as f:weights={key:file.name for key in f.keys()}
    mapping={}
    for key,filename in weights.items():
        name=key[6:] if key.startswith('model.') else key
        if name in expected:
            assert name not in mapping, f'ambiguous key {name}'
            assert Path(filename).name==filename, 'Nonlocal shard path'
            mapping[name]=(filename,key)
    assert set(mapping)==set(expected), {'missing':sorted(set(expected)-set(mapping))}
    started=time.monotonic()
    # Close the mmap after EACH tensor; force a private copy even when source already FP16.
    for i,(name,(filename,key)) in enumerate(sorted(mapping.items())):
        with safe_open(base/filename,framework='pt',device='cpu') as f:
            value=f.get_tensor(key).to(dtype=dtype,device='cpu',copy=True)
            assert tuple(value.shape)==expected[name][0]
        set_module_tensor_to_device(model,name,'cpu',value=value,dtype=dtype)
        del value
        # Private tensor is now installed; advise the kernel that unmapped shard pages are expendable.
        if hasattr(os,'posix_fadvise'):
            with open(base/filename,'rb') as shard:os.posix_fadvise(shard.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        if i%28==0 or i+1==len(mapping):
            gc.collect();state=memory_snapshot()
            print(json.dumps({'event':'stream_load','loaded':i+1,'total':len(mapping),'seconds':time.monotonic()-started,**state}),flush=True)
            assert state['rss_bytes']<27*2**30,'RSS guard during load'
    assert not any(p.is_meta for p in model.parameters())
    assert not any(b.is_meta for b in model.buffers())
    assert all(p.dtype==dtype for p in model.parameters())
    model.eval()
    print(json.dumps({'event':'stream_load_complete','parameters':len(mapping),'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),**memory_snapshot()}),flush=True)
    return model


def validate_sharded_fixture(merge_lora):
    import tempfile
    from safetensors.torch import save_file
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(1901)
    config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=64)
    with tempfile.TemporaryDirectory() as folder:
        base=Path(folder)/'base';adapter=Path(folder)/'adapter';adapter.mkdir()
        full=Qwen3ForCausalLM(config).to(torch.bfloat16).eval()
        full.save_pretrained(base,max_shard_size='16KB',safe_serialization=True);del full;gc.collect()
        assert (base/'model.safetensors.index.json').exists()
        cfg=AutoConfig.from_pretrained(base);cfg.num_hidden_layers=2;cfg.use_cache=False
        if getattr(cfg,'layer_types',None) is not None:cfg.layer_types=cfg.layer_types[:2]
        old=AutoModel.from_pretrained(base,config=cfg,dtype=torch.float16,device_map={'':'cpu'},low_cpu_mem_usage=True).eval();old.norm=nn.Identity()
        new=load_streamed_trunk(base,2,torch.float16)
        old_params=dict(old.named_parameters());new_params=dict(new.named_parameters())
        assert old_params.keys()==new_params.keys()
        for name in old_params:
            assert old_params[name].dtype==new_params[name].dtype
            assert torch.equal(old_params[name],new_params[name]),name
        old_buf=dict(old.named_buffers());new_buf=dict(new.named_buffers())
        assert old_buf.keys()==new_buf.keys()
        for name in old_buf:assert torch.equal(old_buf[name],new_buf[name]),name
        (adapter/'adapter_config.json').write_text(json.dumps({'lora_alpha':8,'r':4}))
        save_file({'base_model.model.layers.0.self_attn.q_proj.lora_A.weight':torch.randn(4,32)*.01,'base_model.model.layers.0.self_attn.q_proj.lora_B.weight':torch.randn(32,4)*.01},adapter/'adapter_model.safetensors')
        merge_lora(old,str(adapter));merge_lora(new,str(adapter))
        for name in old_params:assert torch.equal(old_params[name],dict(new.named_parameters())[name]),name
        with torch.inference_mode():
            ids=torch.tensor([[1,2,3,4]])
            a=old(input_ids=ids,output_hidden_states=True,use_cache=False).last_hidden_state
            b=new(input_ids=ids,output_hidden_states=True,use_cache=False).last_hidden_state
        assert torch.equal(a,b),'fixture output mismatch'
        result={'passed':True,'parameters_equal':len(old_params),'buffers_equal':len(old_buf),'after_original_lora_merge_equal':True,'forward_bitwise_equal':True,'source_dtype':'BF16','loaded_dtype':'FP16','source_layers':3,'retained_layers':2}
        Path('streaming-fixture.json').write_text(json.dumps(result,indent=2));print(json.dumps({'event':'streaming_fixture_pass',**result}),flush=True)
