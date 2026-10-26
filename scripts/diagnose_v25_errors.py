"""Fixed saved-prediction diagnostics, without tuning or new training."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path('artifacts/improvement-v25')
    frame = pd.read_parquet(Path('artifacts/improvement-v22/kaggle-output/validation_frame.parquet'))
    paths = {
        'v22_supcon': Path('artifacts/improvement-v22/kaggle-output/supcon_final_validation.npz'),
        'v25_supcon': root/'quick-results/supcon_final_validation.npz',
        'v25_ce': root/'quick-results/ce_final_validation.npz',
        'sparse': Path('artifacts/improvement-v17/uniform_pool_pilot.npz'),
    }
    scores = {}
    labels = None
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            assert data['ids'].tolist() == frame.id.astype(str).tolist()
            if labels is None:
                labels = data['labels']
            assert np.array_equal(labels, data['labels'])
            scores[name] = data['scores'].copy()
    for name in ['v22_supcon', 'v25_supcon', 'v25_ce']:
        scores[name+'_blend'] = scores['sparse'] + 0.15*scores[name]
    truth = frame.model.to_numpy()
    correct = {name: labels[value.argmax(1)] == truth for name,value in scores.items()}
    groups = frame.group.to_numpy()
    unique = np.unique(groups)
    rng = np.random.default_rng(2501)
    samples = rng.integers(0, len(unique), size=(10000,len(unique)))
    comparisons = {}
    for first, second in [('v22_supcon','v25_supcon'), ('v22_supcon_blend','v25_supcon_blend'),
                          ('v22_supcon','v25_ce'), ('v22_supcon_blend','v25_ce_blend'),
                          ('sparse','v25_ce_blend'), ('sparse','v25_supcon_blend')]:
        delta = correct[second].astype(float)-correct[first].astype(float)
        group_means = np.array([delta[groups==g].mean() for g in unique])
        interval = np.quantile(group_means[samples].mean(1), [0.025,0.975]).tolist()
        comparisons[first+' -> '+second] = dict(gain=float(delta.mean()),
            prompt_bootstrap_95_interval=interval, corrected=int((delta>0).sum()),
            newly_wrong=int((delta<0).sum()), prompts_improved=int((group_means>0).sum()),
            prompts_worsened=int((group_means<0).sum()))
    per_class = {label:{name:float(value[truth==label].mean()) for name,value in correct.items()}
                 for label in labels}
    result = dict(rows=len(frame), prompts=len(unique), accuracy={name:float(c.mean()) for name,c in correct.items()},
        comparisons=comparisons, per_class=per_class,
        input_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
        limitations=['Full V25 checkpoint/source audit pending; validation arrays only are checked here',
                     'Post-hoc diagnostic on repeatedly inspected fold; bootstrap does not remove selection bias',
                     'Sparse blend is not the full public ensemble; no revised weights or model selection',
                     'Equal 18-row prompt sizes permit bootstrap of prompt means'])
    (root/'error-diagnostic.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'accuracy':result['accuracy'],'comparisons':comparisons},indent=2))


if __name__ == '__main__':
    main()
