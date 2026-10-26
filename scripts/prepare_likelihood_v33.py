"""Prepare one fixed host holdout, retaining related-prompt group boundaries."""
import hashlib
import json
from pathlib import Path

import pandas as pd


def prepare(host_path, split_by_prompt, plan, output):
    host_path = Path(host_path)
    assert hashlib.sha256(host_path.read_bytes()).hexdigest() == plan['host_sha256']
    host = pd.read_csv(host_path, dtype={'id': str, 'prompt_id': str})
    assert len(host) == 2699 and host.id.is_unique
    assert set(host.prompt_id) == set(split_by_prompt)
    labels = sorted(host.model.unique())
    assert labels == plan['labels'] and len(labels) == 18
    host['group'] = host.prompt_id.map(lambda p: str(split_by_prompt[p]['group']))
    assigned = host.prompt_id.map(lambda p: int(split_by_prompt[p]['fold']))
    assert host.assign(fold=assigned).groupby('group').fold.nunique().max() == 1
    heldout = assigned == plan['confirmation_fold']
    host['stage'] = 'host'
    host['label'] = host.model.map({name: i for i, name in enumerate(labels)}).astype(int)
    train = host.loc[~heldout].reset_index(drop=True)
    valid = host.loc[heldout].reset_index(drop=True)
    assert len(train) == plan['training_rows'] and len(valid) == plan['validation_rows']
    assert set(train.group).isdisjoint(valid.group)
    assert set(train.prompt_id).isdisjoint(valid.prompt_id)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / name for name in ['training_frame.parquet', 'validation_frame.parquet']]
    assert not any(p.exists() for p in paths), 'Existing prepared frames must not be overwritten'
    for path, frame in zip(paths, [train, valid]):
        frame.to_parquet(path, index=False)
    audit = {'split': plan['confirmation_split'], 'training_rows': len(train),
             'validation_rows': len(valid), 'training_groups': train.group.nunique(),
             'validation_groups': valid.group.nunique(), 'labels': labels,
             'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (output / 'data-audit.json').write_text(json.dumps(audit, indent=2))
    return audit
