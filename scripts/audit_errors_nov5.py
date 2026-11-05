"""Read-only prediction diagnostics. No blend tuning, fitting or public-score claim."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    sources = {}
    def load(relative):
        p = root/relative
        sources[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
        return np.load(p, allow_pickle=False)
    train_path = root/'artifacts/improvement-v42/inputs/train.csv'
    sources[str(train_path.relative_to(root))] = hashlib.sha256(train_path.read_bytes()).hexdigest()
    host = pd.read_csv(train_path, dtype={'id':str, 'prompt_id':str}).set_index('id')
    z = load('artifacts/improvement-v34/primary-oof.npz')
    assert len(set(z['ids'])) == len(z['ids'])
    frame = host.loc[z['ids']]
    labels = z['labels'].tolist()
    y = np.array([labels.index(v) for v in frame.model])
    candidate, comparison = z['candidate_predictions'], z['comparison_predictions']
    raw = z['candidate_scores'].argmax(1)
    assert np.isfinite(z['candidate_scores']).all()
    pairs = Counter(zip(y[candidate != y], candidate[candidate != y]))
    class_rows = []
    for k, name in enumerate(labels):
        mask = y == k
        class_rows.append(dict(label=name, rows=int(mask.sum()), candidate_correct=int((candidate[mask]==k).sum()),
            comparison_correct=int((comparison[mask]==k).sum()), raw_8b_correct=int((raw[mask]==k).sum())))
    topk = {str(k):float(np.mean(np.any(np.argsort(z['candidate_scores'],axis=1)[:,-k:] == y[:,None],axis=1))) for k in [1,2,3]}
    families = np.array([name.split('-')[0] for name in labels])
    wrong = candidate != y
    within = wrong & (families[candidate] == families[y])
    complements = []
    for arm in ['dense_ce','blocks_ce','dense_supcon','blocks_supcon']:
        counts = Counter()
        for fold in [1,2]:
            c = load(f'artifacts/improvement-v42/final-validation/{arm}/fold-{fold}/validation.npz')
            mask = z['fold'] == fold
            assert np.array_equal(c['ids'],z['ids'][mask]) and c['labels'].tolist()==labels
            assert np.array_equal(c['targets'],y[mask])
            a, b, truth = raw[mask], c['logits'].argmax(1), y[mask]
            counts.update(rows=len(truth), base_correct=int((a==truth).sum()), alternative_correct=int((b==truth).sum()),
                base_wrong_alternative_right=int(((a!=truth)&(b==truth)).sum()),
                base_right_alternative_wrong=int(((a==truth)&(b!=truth)).sum()),
                both_wrong=int(((a!=truth)&(b!=truth)).sum()))
        complements.append(dict(arm=arm, **counts))
    report = dict(date='2026-11-05', kind='descriptive_error_audit', source_sha256=sources,
        primary_rows=len(y), primary_candidate_correct=int((candidate==y).sum()),
        primary_comparison_correct=int((comparison==y).sum()),
        candidate_corrects_comparison=int(((candidate==y)&(comparison!=y)).sum()),
        candidate_regresses_comparison=int(((candidate!=y)&(comparison==y)).sum()),
        within_provider_errors=int(within.sum()), cross_provider_errors=int((wrong & ~within).sum()),
        raw_8b_topk_recall=topk, classes=class_rows,
        largest_directed_confusions=[dict(truth=labels[a],predicted=labels[b],count=n) for (a,b),n in pairs.most_common(20)],
        v42_complementarity_vs_raw_8b=complements,
        interpretation='Deliver qualified V39 first. For new experiments, prioritize difficult model pairs and full-ensemble validation over weak global block blends.',
        limitations=['Primary candidate/comparison predictions are sparse+neural+fixed balancing, not the full public .622 ensemble.',
            'Raw neural top-k and oracle complement counts are descriptive ceilings, not attainable scores or proposed prediction rules.',
            'All splits have been inspected repeatedly; selecting a specialist from these errors requires an independent confirmation split.',
            'No new model fitted, no hypothesis tested prospectively, no expected leaderboard gain claimed.'])
    out = root/'artifacts/nov5-error-audit';out.mkdir(exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
