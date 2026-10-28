"""Audit aligned predictions and compute the frozen V40 comparison."""
import argparse
import json
from pathlib import Path
import numpy as np
from model_blocks_v40 import balanced_accuracy


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());root=Path(a.output)
    by_arm={};folds={};reference={}
    for arm in plan['arms']:
        targets=[];predictions=[];folds[arm]={}
        for fold in plan['folds']:
            folder=root/arm/f'fold-{fold}'
            result=json.loads((folder/'result.json').read_text())
            assert result['completed'] and result['changed_adapter_tensors']>0
            with np.load(folder/'validation.npz',allow_pickle=False) as data:
                assert data['labels'].tolist()==plan['labels']
                assert np.isfinite(data['logits']).all()
                assert data['logits'].shape==(len(data['ids']),len(plan['labels']))
                current=(data['ids'].copy(),data['targets'].copy(),data['prompts'].copy())
                if fold not in reference:reference[fold]=current
                assert all(np.array_equal(x,y) for x,y in zip(current,reference[fold]))
                score=balanced_accuracy(data['targets'],data['logits'].argmax(1))
                assert abs(score-result['balanced_accuracy'])<1e-12
                folds[arm][str(fold)]=score
                targets.extend(data['targets']);predictions.extend(data['logits'].argmax(1))
        by_arm[arm]=balanced_accuracy(np.asarray(targets),np.asarray(predictions))
    assert len(set(np.concatenate([v[0] for v in reference.values()])))==sum(len(v[0]) for v in reference.values())
    primary=by_arm['blocks_supcon']-by_arm['dense_supcon']
    report={'pooled_balanced_accuracy':by_arm,'folds':folds,'primary_gain':primary,
        'block_gain_without_supcon':by_arm['blocks_ce']-by_arm['dense_ce'],
        'supcon_gain_dense':by_arm['dense_supcon']-by_arm['dense_ce'],
        'screen_pass':primary>=plan['gate']['blocks_supcon_over_dense_supcon_pooled_min'] and all(folds['blocks_supcon'][str(f)]>folds['dense_supcon'][str(f)] for f in plan['folds']),
        'scope':'Two-fold standalone classifier screening; no leaderboard claim'}
    (root/'screen.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
