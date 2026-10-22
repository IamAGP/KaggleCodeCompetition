"""Synthetic contracts for v8 prompt recovery and within-prompt assignment."""

import numpy as np

from name_that_llm.assignment_v8 import assign_within_prompts, recover_prompts

TOPICS = ["volcano lava eruption magma crater ash", "violin orchestra concerto symphony bow strings",
          "glacier ice arctic melting snow polar", "bitcoin blockchain ledger mining crypto wallet"]


def test_recovers_topic_groups_and_respects_cap():
    texts = [f"{topic} passage variant {i}" for topic in TOPICS for i in range(18)]
    groups = recover_prompts(texts)
    assert np.bincount(groups).max() <= 18
    for t in range(len(TOPICS)):
        assert len(set(groups[t * 18:(t + 1) * 18])) == 1


def test_skips_tiny_inputs():
    assert recover_prompts(["one", "two", "three"]) is None


def test_assignment_is_one_to_one_within_group():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(18, 18))
    scores[:, 3] += 5  # every row prefers class 3 independently
    prediction = assign_within_prompts(scores, np.zeros(18, dtype=int))
    assert sorted(prediction) == list(range(18))
    assert (assign_within_prompts(scores, None) == 3).all()
