"""Coverage-first crossed batches, separate from frozen V35/V40 experiments.

Every batch has two authors answering the same two prompts. Every supplied
training row appears at least once per epoch; missing cells may require repeats.
The caller must pass only the training partition. No validation rows are added.
"""
from collections import Counter
import numpy as np


def coverage_batches(frame, seed):
    if frame.empty or set(frame.stage) != {'host'}:
        raise ValueError('Expected nonempty host training partition')
    if not frame.index.is_unique or frame.duplicated(['label', 'group']).any():
        raise ValueError('Expected unique row IDs and author/prompt cells')
    cells = {(r.label, r.group): i for i, r in frame.iterrows()}
    authors = sorted(frame.label.unique())
    prompts = sorted(frame.group.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(prompts)
    counts = Counter()
    batches = []

    def append(a, b, p, q):
        rows = [cells[a, p], cells[a, q], cells[b, p], cells[b, q]]
        batches.append(rows)
        counts.update(rows)

    # Disjoint rectangles cover a complete even-sized grid exactly once.
    for k in range(0, len(prompts)-1, 2):
        p, q = prompts[k:k+2]
        shared = [a for a in authors if (a, p) in cells and (a, q) in cells]
        rng.shuffle(shared)
        for j in range(0, len(shared)-1, 2):
            append(shared[j], shared[j+1], p, q)

    # Repair odd dimensions and holes using only legal cells in this partition.
    remaining = list(frame.index)
    rng.shuffle(remaining)
    for i in remaining:
        if counts[i]:
            continue
        a, p = frame.loc[i, ['label', 'group']]
        candidates = []
        for b in authors:
            if b == a or (b, p) not in cells:
                continue
            for q in prompts:
                if q == p or (a, q) not in cells or (b, q) not in cells:
                    continue
                ids = [cells[a, p], cells[a, q], cells[b, p], cells[b, q]]
                priority = (sum(counts[x] == 0 for x in ids), -sum(counts[x] for x in ids))
                candidates.append((priority, b, q))
        if not candidates:
            raise ValueError(f'Row {i!r} cannot form a crossed batch within training partition')
        best = max(x[0] for x in candidates)
        tied = [x for x in candidates if x[0] == best]
        _, b, q = tied[int(rng.integers(len(tied)))]
        append(a, b, p, q)
    assert set(counts) == set(frame.index)
    rng.shuffle(batches)
    return batches
