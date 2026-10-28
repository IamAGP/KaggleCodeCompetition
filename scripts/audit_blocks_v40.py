"""Independent V40 prediction audit and prompt-cluster uncertainty; no tuning."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.root);plan=json.loads(Path(args.plan).read_text())
    references={};correct={};audits=[]
    for arm in plan['arms']:
        correct[arm]=[]
        for fold in plan['folds']:
            folder=root/arm/f'fold-{fold}'
            config=json.loads((folder/'run-config.json').read_text())
            result=json.loads((folder/'result.json').read_text())
            assert config==dict(plan=plan,arm=arm,fold=fold)
            assert result['completed'] and result['changed_adapter_tensors']==144
            with np.load(folder/'validation.npz',allow_pickle=False) as d:
                assert d['labels'].tolist()==plan['labels']
                assert d['logits'].shape==(540,18) and np.isfinite(d['logits']).all()
                assert len(set(d['ids']))==540
                assert np.all(np.bincount(d['targets'],minlength=18)==30)
                key=(d['ids'].tolist(),d['targets'].tolist(),d['prompts'].tolist())
                references.setdefault(fold,key);assert key==references[fold]
                hits=d['logits'].argmax(1)==d['targets'];score=float(hits.mean())
                assert abs(score-result['balanced_accuracy'])<1e-12
                correct[arm].append(hits)
                audits.append({'arm':arm,'fold':fold,'balanced_accuracy':score,'correct':int(hits.sum())})
    first,second=plan['folds'];assert set(references[first][0]).isdisjoint(references[second][0])
    assert set(references[first][2]).isdisjoint(references[second][2])
    prompt_ids=np.concatenate([references[f][2] for f in plan['folds']])
    hits={a:np.concatenate(parts) for a,parts in correct.items()}
    differences=hits['blocks_supcon'].astype(float)-hits['dense_supcon']
    prompts=np.unique(prompt_ids)
    assert len(prompts)==60 and all(np.sum(prompt_ids==p)==18 for p in prompts)
    prompt_differences=np.array([differences[prompt_ids==p].mean() for p in prompts])
    rng=np.random.default_rng(4001)
    bootstrap=prompt_differences[rng.integers(0,len(prompts),size=(10000,len(prompts)))].mean(1)
    gains=[float((b.astype(float)-a).mean()) for a,b in zip(correct['dense_supcon'],correct['blocks_supcon'])]
    primary=float(differences.mean())
    report={'audits':audits,'pooled':{a:float(h.mean()) for a,h in hits.items()},
        'primary_gain':primary,'primary_fold_gains':gains,
        'primary_prompt_bootstrap_95_percentile_interval':np.quantile(bootstrap,[.025,.975]).tolist(),
        'bootstrap_scope':'Descriptive uncertainty across 60 held-out prompt groups; previously inspected splits, not untouched confirmation.',
        'frozen_screen_pass':primary>=plan['gate']['blocks_supcon_over_dense_supcon_pooled_min'] and all(g>0 for g in gains),
        'scope':'Standalone V40 comparison, not the incumbent ensemble or Kaggle score. No fitted weights or threshold search.'}
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
