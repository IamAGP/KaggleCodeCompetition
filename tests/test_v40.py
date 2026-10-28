"""Scientific implementation checks for V40, without GPU/model downloads."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
import pytest
import torch
from model_blocks_v40 import AttributionHeads, BlockReadout, masked_mean, supervised_contrastive
from train_adapter_v18 import supcon_loss
from sampling_v35 import crossed_batches


def test_only_assigned_block_changes_its_class_score():
    head=BlockReadout(3,4)
    z=torch.randn(2,12,requires_grad=True)
    head(z)[:,1].sum().backward()
    assert torch.count_nonzero(z.grad[:,:4])==0
    assert torch.count_nonzero(z.grad[:,8:])==0
    assert torch.count_nonzero(z.grad[:,4:8])>0


def test_classification_updates_representation_and_input():
    head=AttributionHeads(8,True,3,4)
    x=torch.randn(6,8,requires_grad=True)
    scores,projection,z=head(x)
    loss=torch.nn.functional.cross_entropy(scores,torch.tensor([0,0,1,1,2,2]))
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert head.representation[0].weight.grad.abs().sum()>0
    assert z.shape==(6,12) and projection.shape==(6,128)


def test_supcon_matches_audited_loss_and_is_permutation_invariant():
    features=torch.randn(6,12,requires_grad=True);labels=torch.tensor([0,0,1,1,2,2])
    loss=supervised_contrastive(features,labels)
    torch.testing.assert_close(loss,supcon_loss(features,labels))
    ix=torch.tensor([4,1,3,0,5,2])
    torch.testing.assert_close(loss,supervised_contrastive(features[ix],labels[ix]))
    loss.backward();assert torch.isfinite(features.grad).all()
    with pytest.raises(ValueError):supervised_contrastive(features,torch.arange(6))


def test_pooling_ignores_padding_and_keeps_first_word():
    h=torch.tensor([[[2.,4.],[6.,8.],[999.,999.]]])
    torch.testing.assert_close(masked_mean(h,torch.tensor([[1,1,0]])),torch.tensor([[4.,6.]]))


def test_shared_initialization_matches_across_heads():
    torch.manual_seed(4001);a=AttributionHeads(8,False,3,4)
    torch.manual_seed(4001);b=AttributionHeads(8,True,3,4)
    for module in ['norm','representation','projection']:
        for key,value in getattr(a,module).state_dict().items():
            torch.testing.assert_close(value,getattr(b,module).state_dict()[key],rtol=0,atol=0)


def test_crossed_batches_have_positive_and_matched_negative():
    frame=pd.DataFrame([dict(label=m,group=p,stage='host') for m in range(3) for p in range(4)])
    batches=list(crossed_batches(frame,10,4001))
    assert batches==list(crossed_batches(frame,10,4001))
    for ids in batches:
        rows=frame.loc[ids]
        assert len(set(ids))==4 and rows.label.nunique()==2 and rows.group.nunique()==2
        assert rows.groupby('label').group.nunique().tolist()==[2,2]
        assert rows.groupby('group').label.nunique().tolist()==[2,2]
