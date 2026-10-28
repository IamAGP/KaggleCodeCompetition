import sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pandas as pd
import pytest
from sampling_v42 import coverage_batches


def grid(authors, prompts):
    return pd.DataFrame([{'label':a, 'group':p, 'stage':'host'}
                         for a in range(authors) for p in range(prompts)])


@pytest.mark.parametrize('authors,prompts,hole', [(18,120,False),(18,120,True),(3,5,False)])
def test_full_coverage_with_valid_contrastive_pairs(authors, prompts, hole):
    frame = grid(authors, prompts)
    if hole:
        frame = frame.drop(index=9)
    batches = coverage_batches(frame, 4201)
    assert batches == coverage_batches(frame, 4201)
    counts = Counter(i for batch in batches for i in batch)
    assert set(counts) == set(frame.index)
    for batch in batches:
        rows = frame.loc[batch]
        assert len(set(batch)) == 4
        assert rows.label.nunique() == rows.group.nunique() == 2
        assert rows.groupby('label').group.nunique().eq(2).all()
    if authors % 2 == prompts % 2 == 0 and not hole:
        assert set(counts.values()) == {1}
    if hole:
        assert len(batches) <= 543


def test_does_not_repair_with_held_out_prompts():
    frame = grid(4,6)
    fit = frame.loc[frame.group < 3]
    assert all(set(batch) <= set(fit.index) for batch in coverage_batches(fit, 4201))


def test_unpairable_row_fails_without_silently_dropping_it():
    frame = grid(2,2).drop(index=0)
    with pytest.raises(ValueError, match='cannot form'):
        coverage_batches(frame, 4201)
