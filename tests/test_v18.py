"""Check contrastive learning semantics before spending GPU training time."""
import importlib.util
from pathlib import Path
import pandas as pd
import pytest
import torch

spec = importlib.util.spec_from_file_location("v18_worker", Path(__file__).parents[1] / "scripts/train_adapter_v18.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_cross_prompt_pairs_and_reproducible_control_schedule():
    frame = pd.DataFrame({"label":[0,0,0,1,1,1], "group":["a","a","b","a","b","b"]})
    batches = list(worker.paired_batches(frame, 30, 1801))
    assert batches == list(worker.paired_batches(frame, 30, 1801))
    for batch in batches:
        assert len(set(batch)) == 4
        chosen = frame.loc[batch]
        assert chosen.groupby("label").size().tolist() == [2,2]
        assert chosen.groupby("label").group.nunique().tolist() == [2,2]


def test_contrastive_loss_rewards_author_separation_and_has_finite_gradients():
    labels = torch.tensor([0,0,1,1])
    aligned = torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]], requires_grad=True)
    mixed = torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
    loss = worker.supcon_loss(aligned, labels)
    assert loss < worker.supcon_loss(mixed, labels)
    loss.backward()
    assert torch.isfinite(aligned.grad).all()
    permutation = torch.tensor([2,0,3,1])
    assert torch.allclose(loss, worker.supcon_loss(aligned[permutation], labels[permutation]))


def test_no_self_pair_can_replace_missing_positive():
    with pytest.raises(ValueError, match="distinct positive"):
        worker.supcon_loss(torch.eye(4), torch.arange(4))
