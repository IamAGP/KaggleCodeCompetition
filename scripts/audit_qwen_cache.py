"""Real Qwen3/PEFT/checkpoint API integration gate, tiny random CPU model.

No pretrained weights or competition data; does not certify CUDA NF4 numerics.
"""
import copy
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3Model
from peft import LoraConfig, get_peft_model
from contrastive_cache import cached_backward
from model_blocks_v40 import AttributionHeads, masked_mean, supervised_contrastive


def run(blocked, bf16):
    torch.manual_seed(4602)
    config = Qwen3Config(vocab_size=100, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, use_cache=False,
                        attention_dropout=0.1)
    model = get_peft_model(Qwen3Model(config), LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.1, target_modules=['q_proj', 'v_proj']))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train()
    heads = AttributionHeads(32, blocked)
    ref_model, ref_heads = copy.deepcopy(model), copy.deepcopy(heads)
    batches = [{'input_ids': torch.randint(1, 100, (4, 12)),
                'attention_mask': torch.ones(4, 12, dtype=torch.long)} for _ in range(3)]
    batches[-1]['attention_mask'][:, -3:] = 0
    labels = torch.arange(6).repeat_interleave(2)

    def forward(m, h, batch):
        with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
            hidden = m(**batch, use_cache=False).last_hidden_state
            scores, projection, _ = h(masked_mean(hidden, batch['attention_mask']))
        return torch.cat([scores.float(), projection.float()], dim=1)

    def loss(output):
        return F.cross_entropy(output[:, :18], labels) + 0.1 * supervised_contrastive(output[:, 18:], labels)

    def grads(m, h):
        return {name: p.grad.clone() for prefix, module in [('encoder', m), ('heads', h)]
                for short, p in module.named_parameters() if p.requires_grad
                for name in [prefix+'.'+short]}

    initial = torch.get_rng_state()
    reference_loss = loss(torch.cat([forward(ref_model, ref_heads, b) for b in batches]))
    reference_loss.backward()
    expected_rng = torch.get_rng_state()
    expected = grads(ref_model, ref_heads)
    torch.set_rng_state(initial)
    actual_loss = cached_backward(lambda b: forward(model, heads, b), batches, loss)
    actual = grads(model, heads)
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=2e-4, atol=2e-6, msg=name)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(actual_loss, reference_loss)
    return dict(blocked=blocked, bf16=bf16, tensors_checked=len(actual),
                max_gradient_error=max(float((actual[k]-expected[k]).abs().max()) for k in actual),
                checkpointing=True, lora_dropout=0.1, passed=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    result = {'status': 'passed', 'scope': 'Tiny random Qwen3, CPU; CUDA NF4 not tested',
              'cases': [run(b, f) for b in (False, True) for f in (False, True)]}
    Path('configs/name-that-llm/qwen-cache-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
