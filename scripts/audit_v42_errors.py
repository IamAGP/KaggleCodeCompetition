"""Describe paired V42 class errors; never fit weights or select a checkpoint."""
import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / 'configs/name-that-llm/v42.json').read_text())
    labels = plan['labels']
    n = len(labels)
    matrices, hits = {}, {}
    refs = {}
    for arm in plan['arms']:
        matrix = np.zeros((n, n), dtype=int)
        hits[arm] = []
        for fold in plan['folds']:
            folder = root / 'artifacts/improvement-v42/final-validation' / arm / f'fold-{fold}'
            assert json.loads((folder / 'run-config.json').read_text()) == dict(plan=plan, arm=arm, fold=fold)
            z = np.load(folder / 'validation.npz', allow_pickle=False)
            assert z['labels'].tolist() == labels
            assert len(set(z['ids'])) == len(z['ids'])
            assert np.isfinite(z['logits']).all()
            key = tuple(tuple(z[k].tolist()) for k in ['ids', 'targets', 'prompts'])
            refs.setdefault(fold, key)
            assert refs[fold] == key
            y, p = z['targets'], z['logits'].argmax(1)
            assert np.all(np.bincount(y, minlength=n) == 30)
            np.add.at(matrix, (y, p), 1)
            hits[arm].extend((p == y).tolist())
        matrices[arm] = matrix
    assert set(refs[1][0]).isdisjoint(refs[2][0])
    assert set(refs[1][2]).isdisjoint(refs[2][2])
    a, b = matrices['dense_ce'], matrices['blocks_supcon']
    per_class = [dict(model=label, rows=int(a[i].sum()), dense_correct=int(a[i,i]),
        blocks_contrastive_correct=int(b[i,i]), change=int(b[i,i]-a[i,i])) for i, label in enumerate(labels)]
    confusions = {}
    for arm, m in matrices.items():
        pairs = [(int(m[i,j]), labels[i], labels[j]) for i in range(n) for j in range(n) if i != j and m[i,j]]
        confusions[arm] = [dict(count=c, actual=a, predicted=b) for c,a,b in sorted(pairs, reverse=True)[:10]]
    dense, blocks = np.array(hits['dense_ce']), np.array(hits['blocks_supcon'])
    report = dict(experiment='V42 descriptive error audit', per_class=per_class, top_confusions=confusions,
        blocks_vs_dense=dict(corrected=int((~dense & blocks).sum()), regressed=int((dense & ~blocks).sum())),
        confusion_matrices={k:v.tolist() for k,v in matrices.items()}, labels=labels,
        scope='Previously inspected held-out prompts; descriptive diagnosis only, no fitted classifier or public-score claim.')
    output = root / 'configs/name-that-llm/v42-error-audit.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), blocks_vs_dense=report['blocks_vs_dense'], per_class=per_class)))


if __name__ == '__main__':
    main()
