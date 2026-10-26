"""Semantic checks before running source-conditioned likelihood on GPU."""
import importlib.util
from pathlib import Path
import sys

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

spec = importlib.util.spec_from_file_location('source_likelihood',Path(__file__).parents[1]/'scripts/source_likelihood.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_route_changes_only_selected_source_gradients():
    torch.manual_seed(29)
    route = module.SourceRoute(torch.tensor([0,2]))
    model = module.SourceProjection(torch.nn.Linear(5,7),route,3,rank=2,dropout=0)
    x = torch.randn(2,4,5)
    mixed = model(x)
    for index, source in enumerate([0,2]):
        route.ids = torch.tensor([source])
        torch.testing.assert_close(model(x[index:index+1]),mixed[index:index+1])
    mixed.square().mean().backward()
    assert model.base.weight.grad is None
    assert model.up.grad[1].count_nonzero() == 0
    assert model.up.grad[0].abs().sum()>0 and model.up.grad[2].abs().sum()>0
    assert model.down.grad.abs().sum()>0


def test_clipped_evidence_order_mask_and_shift_invariance():
    lp = torch.tensor([[[-1.,-2.,100.],[-3.,-4.,-100.],[-2.,-3.,0.]]])
    mask = torch.tensor([[1.,1.,0.]])
    scores = module.clipped_evidence(lp,mask)
    assert scores.argmax(1).item() == 0
    assert scores[0,0] == 1 and scores[0,2] > scores[0,1]
    torch.testing.assert_close(scores,module.clipped_evidence(lp+torch.tensor([[[7.,-9.,30.]]]),mask))
    torch.testing.assert_close(scores,module.clipped_evidence(lp[:,:,:2],mask[:,:2]))


def test_causal_alignment_padding_checkpoint_gradients_and_reload():
    torch.manual_seed(2901)
    config = Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=32,
                         num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1)
    model = module.SourceLanguageModel(Qwen2ForCausalLM(config),3,rank=2,dropout=0,head_chunk=2)
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    ids = torch.tensor([[1,4,5,6],[1,7,8,0]])
    mask = torch.tensor([[1,1,1,1],[1,1,1,0]])
    source = torch.tensor([0,2])
    model.train()
    loss = model(ids,mask,source)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.prefix.grad[1].count_nonzero()==0
    for name,p in model.named_parameters():
        if p.requires_grad: assert p.grad is not None and torch.isfinite(p.grad).all(),name
        else: assert p.grad is None,name
    model.eval()
    lp = model.token_log_probs(ids,mask,source)
    # Independent full-vocabulary cross entropy verifies one-token causal shift.
    model.route.ids = source
    inputs = torch.cat([model.prefix[source,None,:],model.lm.get_input_embeddings()(ids)],1)
    hidden = model.lm.get_decoder()(inputs_embeds=inputs,
                attention_mask=torch.cat([torch.ones_like(mask[:,:1]),mask],1),use_cache=False).last_hidden_state
    logits = model.lm.get_output_embeddings()(hidden[:,1:-1])
    expected = logits.log_softmax(-1).gather(-1,ids[:,1:,None]).squeeze(-1)
    torch.testing.assert_close(lp,expected)
    shorter = model.token_log_probs(ids[1:,:3],mask[1:,:3],source[1:])
    torch.testing.assert_close(lp[1:,:2],shorter)
    saved = model.adapter_state()
    original = model.all_source_scores(ids,mask)['clipped']
    with torch.no_grad(): model.prefix.add_(1)
    model.load_adapter_state(saved)
    torch.testing.assert_close(model.all_source_scores(ids,mask)['clipped'],original)
