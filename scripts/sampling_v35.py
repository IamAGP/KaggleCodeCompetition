"""Host-only crossed-author/prompt batches; V35 extension, not original SupCon."""
import numpy as np


def crossed_batches(frame, steps, seed):
    if set(frame.stage) != {'host'}:
        raise ValueError('Crossed sampling requires verified host prompt IDs')
    if not frame.index.is_unique or frame.duplicated(['label', 'group']).any():
        raise ValueError('Expected exactly one response per author/prompt cell')
    groups = {label: {p: int(sub.index[0]) for p, sub in group.groupby('group')}
              for label, group in frame.groupby('label')}
    labels = sorted(groups)
    if len(labels) < 2 or any(len(set(groups[a]) & set(groups[b])) < 2
                            for a in labels for b in labels if a != b):
        raise ValueError('Every author pair needs two shared prompt IDs')
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        a, b = rng.choice(labels, 2, replace=False)
        shared = sorted(set(groups[a]) & set(groups[b]))
        p, q = rng.choice(shared, 2, replace=False)
        yield [groups[a][p], groups[a][q], groups[b][p], groups[b][q]]
