"""Independent implementation of source-conditioned likelihood modeling.

Method credit: Ko et al., ATLAS, arXiv:2610.19583, equations in sections 3/B.1.
Authors' code reviewed at 93c1b15da6143161e222eecee7ebcdc3c60645a9;
no source copied. This module uses plain PyTorch adapters, not PEFT internals.
Routing must stay unchanged until backward completes (including recomputation).
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class SourceRoute:
    ids: torch.Tensor | None = None


class SourceProjection(nn.Module):
    def __init__(self, linear, route, sources, rank=16, alpha=32, dropout=.05):
        super().__init__()
        self.base = linear.requires_grad_(False)
        self.route = route
        self.factor = alpha/rank
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Parameter(torch.empty(rank, linear.in_features, device=linear.weight.device))
        self.up = nn.Parameter(torch.empty(sources, rank, linear.out_features, device=linear.weight.device))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        nn.init.normal_(self.up, std=1e-4)

    def forward(self, x):
        ids = self.route.ids
        if ids is None or ids.shape != (len(x),):
            raise ValueError('Missing or stale source routing')
        output = self.base(x)
        projected = F.linear(self.dropout(x.to(self.down.dtype)), self.down)
        update = torch.bmm(projected, self.up[ids])
        return output + (self.factor*update).to(output.dtype)


def clipped_evidence(log_probs, mask, temperatures=(1., 2., 4., 8.)):
    """[batch, source, token] -> [batch, source]; padding contributes nothing."""
    if log_probs.ndim != 3 or mask.shape != log_probs[:,0].shape:
        raise ValueError('Invalid token evidence dimensions')
    if (mask.sum(-1) == 0).any():
        raise ValueError('Every passage needs a scored token')
    lp = log_probs.float()
    disadvantage = (lp[:,None,:,:]-lp[:,:,None,:]).clamp_min(0).sum(2)
    evidence = torch.stack([torch.exp(-disadvantage/t) for t in temperatures]).mean(0)
    return (evidence*mask[:,None,:]).sum(-1)/mask.sum(-1)[:,None]


class SourceLanguageModel(nn.Module):
    def __init__(self, causal_lm, sources, rank=16, alpha=32, dropout=.05, head_chunk=32):
        super().__init__()
        self.lm = causal_lm.requires_grad_(False)
        self.sources, self.head_chunk = sources, head_chunk
        self.route = SourceRoute()
        self.lm.config.use_cache = False
        names = [name for name, module in self.lm.named_modules()
                 if isinstance(module, nn.Linear) and name.rsplit('.',1)[-1] in
                 {'q_proj','k_proj','v_proj','o_proj'}]
        if len(names) != 4*self.lm.config.num_hidden_layers:
            raise ValueError('Unexpected attention projection layout')
        for name in names:
            parent_name, attr = name.rsplit('.',1)
            parent = self.lm.get_submodule(parent_name)
            setattr(parent, attr, SourceProjection(getattr(parent, attr),self.route,
                                                  sources,rank,alpha,dropout))
        embedding = self.lm.get_input_embeddings().weight.detach().float()
        initial = embedding.mean(0)[None,:].expand(sources,-1).clone()
        initial += .01*embedding.std()*torch.randn_like(initial)
        self.prefix = nn.Parameter(initial)

    def token_log_probs(self, token_ids, mask, sources):
        if sources.shape != (len(token_ids),) or ((sources<0)|(sources>=self.sources)).any():
            raise ValueError('Invalid source IDs')
        if token_ids.shape != mask.shape or token_ids.shape[1] < 2:
            raise ValueError('Invalid token batch')
        self.route.ids = sources
        embedding = self.lm.get_input_embeddings()(token_ids)
        prefix = self.prefix[sources,None,:].to(embedding.dtype)
        inputs = torch.cat((prefix,embedding),dim=1)
        attention = torch.cat((torch.ones_like(mask[:,:1]),mask),dim=1)
        hidden = self.lm.get_decoder()(inputs_embeds=inputs,attention_mask=attention,
                                      use_cache=False).last_hidden_state[:,1:-1,:]
        target = token_ids[:,1:]

        def score_chunk(h, target_chunk):
            logits = self.lm.get_output_embeddings()(h).float()
            return -F.cross_entropy(logits.flatten(0,1),target_chunk.flatten(),
                                    reduction='none').view_as(target_chunk)

        parts = []
        for start in range(0,hidden.shape[1],self.head_chunk):
            h, target_chunk = hidden[:,start:start+self.head_chunk],target[:,start:start+self.head_chunk]
            value = checkpoint(score_chunk,h,target_chunk,use_reentrant=False) if self.training else score_chunk(h,target_chunk)
            parts.append(value)
        return torch.cat(parts,dim=1)

    def forward(self, token_ids, mask, sources):
        lp = self.token_log_probs(token_ids,mask,sources)
        valid = mask[:,1:].float()
        if (valid.sum(-1)==0).any(): raise ValueError('Empty scored passage')
        return -((lp*valid).sum(-1)/valid.sum(-1)).mean()

    @torch.no_grad()
    def all_source_scores(self, token_ids, mask):
        if self.training: raise ValueError('Call eval before scoring all sources')
        lp = torch.stack([self.token_log_probs(token_ids,mask,
                         torch.full((len(token_ids),),k,device=token_ids.device,dtype=torch.long))
                         for k in range(self.sources)],dim=1)
        valid = mask[:,1:].float()
        mean = (lp*valid[:,None,:]).sum(-1)/valid.sum(-1)[:,None]
        return {'mean':mean,'clipped':clipped_evidence(lp,valid),'token_log_probs':lp}

    def adapter_state(self):
        return {name:parameter.detach().cpu().clone() for name,parameter in self.named_parameters()
                if parameter.requires_grad}

    def load_adapter_state(self, state):
        expected = {name:p for name,p in self.named_parameters() if p.requires_grad}
        if set(state) != set(expected): raise ValueError('Adapter keys differ')
        with torch.no_grad():
            for name, value in state.items():
                if value.shape != expected[name].shape or not torch.isfinite(value).all():
                    raise ValueError('Invalid adapter tensor: '+name)
                expected[name].copy_(value)
