"""V40: shared nonlinear representation with dense or class-block readout.

SupCon follows Khosla et al. (2004.11362), via the audited V18 loss.
Separate class blocks are this experiment's intervention, not Matryoshka MRL.
"""
import torch
from torch import nn
from torch.nn import functional as F


class BlockReadout(nn.Module):
    def __init__(self, classes=18, block_width=100):
        super().__init__()
        self.classes, self.block_width = classes, block_width
        self.weight = nn.Parameter(torch.empty(classes, block_width))
        self.bias = nn.Parameter(torch.zeros(classes))
        nn.init.normal_(self.weight, std=block_width ** -0.5)

    def forward(self, z):
        z = z.reshape(-1, self.classes, self.block_width)
        return (z * self.weight[None]).sum(-1) + self.bias


class AttributionHeads(nn.Module):
    def __init__(self, encoder_width, blocked, classes=18, block_width=100):
        super().__init__()
        width = classes * block_width
        self.norm = nn.LayerNorm(encoder_width)
        self.representation = nn.Sequential(nn.Linear(encoder_width, width), nn.GELU())
        # Construct common modules before the differing head: identical initial tensors.
        self.projection = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 128))
        self.classifier = (BlockReadout(classes, block_width) if blocked
                           else nn.Linear(width, classes))

    def forward(self, pooled):
        z = self.representation(self.norm(pooled.float()))
        return self.classifier(z), self.projection(z), z


def masked_mean(hidden, attention_mask):
    """Mean of all real tokens, including the appended EOS; never discard word 1."""
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    if bool((mask.sum(1) == 0).any()):
        raise ValueError('Empty passage attention mask')
    return (hidden.float() * mask).sum(1) / mask.sum(1)


def supervised_contrastive(features, labels, temperature=0.1):
    z = F.normalize(features.float(), dim=1)
    logits = z @ z.T / temperature
    diagonal = torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    counts = positives.sum(1)
    if not bool((counts > 0).all()):
        raise ValueError('Every anchor needs a different same-author positive')
    denominator = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
    return -((logits - denominator[:, None]).masked_fill(~positives, 0).sum(1) / counts).mean()


def balanced_accuracy(targets, predictions, classes=18):
    values = []
    for label in range(classes):
        mask = targets == label
        if not mask.any():
            raise ValueError(f'Missing evaluation class {label}')
        values.append(float((predictions[mask] == label).mean()))
    return sum(values) / len(values)
