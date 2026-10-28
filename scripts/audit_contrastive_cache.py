"""CPU engineering gate for a possible larger-batch experiment; no training claim.

Method: Gao et al., https://arxiv.org/abs/2101.06983 (section 3.3).
Reviewed author implementation: https://github.com/luyug/GradCache,
src/grad_cache/{grad_cache,context_managers}.py. Independent small audit;
not a GPU/BF16/quantized implementation or a vendored package.
"""
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from model_blocks_v40 import AttributionHeads, supervised_contrastive


class AuditModel(nn.Module):
    def __init__(self, blocked, dropout):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(12, 32), nn.GELU(), nn.Dropout(dropout))
        self.heads = AttributionHeads(32, blocked)

    def forward(self, x):
        scores, projection, _ = self.heads(self.encoder(x))
        return torch.cat((scores, projection), dim=1)


def objective(output, labels):
    return F.cross_entropy(output[:, :18], labels) + 0.1 * supervised_contrastive(
        output[:, 18:], labels, temperature=0.1)


def gradients(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def run(blocked, dropout):
    torch.manual_seed(4601)
    model = AuditModel(blocked, dropout)
    x, labels = torch.randn(36, 12), torch.arange(18).repeat_interleave(2)
    chunks = list(x.split(4))
    initial_rng = torch.get_rng_state()
    reference = copy.deepcopy(model)
    # One joint loss across all 36 rows. Chunked forwards keep identical dropout
    # draws to the cache algorithm; all reference activation graphs remain live.
    ref_output = torch.cat([reference(chunk) for chunk in chunks])
    ref_loss = objective(ref_output, labels)
    ref_loss.backward()
    ref_grad = gradients(reference)
    expected_rng = torch.get_rng_state()

    torch.set_rng_state(initial_rng)
    states, outputs = [], []
    with torch.no_grad():
        for chunk in chunks:
            states.append(torch.get_rng_state())
            outputs.append(model(chunk))
    representation = torch.cat(outputs).detach().requires_grad_()
    loss = objective(representation, labels)
    cached_grad, = torch.autograd.grad(loss, representation)
    # Reproduce dropout during replay without advancing the caller's RNG twice.
    for chunk, state, gradient in zip(chunks, states, cached_grad.split(4)):
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(state)
            replay = model(chunk)
            torch.autograd.backward(replay, gradient)
    actual_grad = gradients(model)
    torch.testing.assert_close(actual_grad, ref_grad, atol=2e-6, rtol=2e-4)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    torch.testing.assert_close(loss, ref_loss)

    # Negative control: ordinary accumulation has the same optimizer population
    # but only two identities per contrastive loss, as in V42.
    accumulated = copy.deepcopy(reference)
    accumulated.zero_grad(set_to_none=True)
    torch.set_rng_state(initial_rng)
    for chunk, y in zip(chunks, labels.split(4)):
        (objective(accumulated(chunk), y) / len(chunks)).backward()
    gap = float((gradients(accumulated) - ref_grad).norm() / ref_grad.norm())
    assert gap > 1e-3, 'Negative control failed to distinguish the objectives'
    return dict(blocked=blocked, dropout=dropout, passed=True,
                cache_max_absolute_gradient_error=float((actual_grad-ref_grad).abs().max()),
                accumulated_relative_gradient_difference=gap,
                rng_state_preserved=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    result = dict(status='cpu_gradient_gate_passed', torch_version=torch.__version__,
                  comparison_rows=36, classes=18, physical_chunk=4,
                  cases=[run(b, d) for b in (False, True) for d in (0.0, 0.2)],
                  scope='Synthetic inputs; actual attribution heads and loss. No accuracy or GPU claim.',
                  next_gate='Validate quantized encoder, BF16, checkpointing and device RNG replay before training.')
    path = Path('configs/name-that-llm/contrastive-cache-audit.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
