"""Audit conditional V34 related confirmation with the predeclared sparse recipe."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '2'
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from build_related_v34 import require_primary
from diagnose_balance_transfer import balance
from evaluate_v34 import audit_fold, balanced_accuracy, sha


def sparse_reference(root, host, heldout, labels):
    """One fixed fit; refuse incomplete cached artifacts rather than silently rerun."""
    path = root / 'sparse-validation.npz'
    receipt = root / 'sparse-result.json'
    spec_path = Path('configs/name-that-llm/v32.json')
    spec = json.loads(spec_path.read_text())
    for name in ['host', 'pool', 'selected_pool_ids']:
        assert sha(spec['sources'][name]) == spec['input_sha256'][name]
    inputs = {'spec_sha256': sha(spec_path), 'worker_sha256': sha(__file__),
              'related_plan_sha256': sha(root / 'frozen-plan.json')}
    valid = host.loc[heldout]
    if path.exists() or receipt.exists():
        assert path.exists() and receipt.exists(), 'Partial sparse fit requires inspection'
        record = json.loads(receipt.read_text())
        assert record['inputs'] == inputs and record['scores_sha256'] == sha(path)
        with np.load(path, allow_pickle=False) as data:
            assert np.array_equal(data['ids'], valid.id) and np.array_equal(data['labels'], labels)
            assert data['scores'].shape == (len(valid), len(labels)) and np.isfinite(data['scores']).all()
            return data['scores'].copy()
    selected = pd.read_csv(spec['sources']['selected_pool_ids'], dtype=str)
    pool = pd.read_parquet(spec['sources']['pool'])
    pool.id = pool.id.astype(str)
    assert pool.id.is_unique
    pool = pool.set_index('id').loc[selected.id].reset_index()
    assert pool[['id', 'source', 'model']].astype(str).equals(selected)
    train = host.loc[~heldout]
    assert set(train.prompt_id).isdisjoint(valid.prompt_id)
    fitting = pd.concat([train, pool], ignore_index=True)
    begin = time.monotonic()
    blocks, test = [], []
    for analyzer, ngram, cap in [('char', (3, 5), 50000), ('word', (1, 2), 30000)]:
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, max_features=cap,
            min_df=2, lowercase=False, sublinear_tf=True, dtype=np.float32)
        blocks.append(vectorizer.fit_transform(fitting.text))
        test.append(vectorizer.transform(valid.text))
    weights = np.r_[np.ones(len(train)), np.full(len(pool), spec['external_weight'])]
    model = LinearSVC(C=spec['C'], class_weight=spec['class_weight'], max_iter=spec['max_iter'],
                      random_state=spec['seed'], dual='auto')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(hstack(blocks).tocsr(), fitting.model, sample_weight=weights)
    assert np.array_equal(model.classes_, labels)
    scores = model.decision_function(hstack(test).tocsr())
    assert np.isfinite(scores).all()
    np.savez_compressed(path, ids=valid.id.to_numpy(dtype=str), labels=labels.astype(str), scores=scores)
    record = {'inputs': inputs, 'scores_sha256': sha(path), 'seconds': time.monotonic()-begin,
              'iterations': int(model.n_iter_), 'fit_host_rows': len(train), 'fit_pool_rows': len(pool),
              'validation_rows': len(valid), 'submission_authorized': False}
    receipt.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    return scores


def score_related(truth, sparse, candidate, comparison, protocol, classes=18):
    assert sparse.shape == candidate.shape == comparison.shape == (len(truth), classes)
    metrics, predictions = {}, {}
    for name, neural in [('candidate', candidate), ('comparison', comparison)]:
        assert np.isfinite(sparse).all() and np.isfinite(neural).all()
        scores = balance(softmax(sparse + protocol['scoring']['component_weight'] * neural, axis=1))
        assert np.isfinite(scores).all() and np.allclose(scores.sum(1), 1)
        predictions[name] = scores.argmax(1)
        metrics[name] = balanced_accuracy(truth, predictions[name], classes)
    gain = metrics['candidate'] - metrics['comparison']
    gate = protocol['related_gate']
    passed = (metrics['candidate'] >= gate['candidate_balanced_accuracy_minimum']
              and gain >= gate['candidate_minus_comparison_minimum'])
    return {'balanced_accuracy': metrics, 'gain': gain, 'qualifies_for_full_host_training': bool(passed),
            'submission_authorized': False}, predictions


def main():
    parent = Path('artifacts/improvement-v34')
    primary = require_primary(parent)
    root = parent / 'related'; out = root / 'kaggle-output'
    frozen = json.loads((root / 'frozen-plan.json').read_text())
    assert sha(parent / 'primary-audit.json') == frozen['primary_audit_sha256']
    assert sha(parent / 'primary-oof.npz') == frozen['primary_oof_sha256']
    remote = json.loads((out / 'plan.json').read_text())
    for key, value in frozen.items():
        if key not in ['notebook_sha256', 'arms']:
            assert remote[key] == value, key
    for arm, spec in frozen['arms'].items():
        assert {k: v for k, v in remote['arms'][arm].items() if k not in ['model_dir', 'checkpoint']} == spec
    for name, digest in frozen['source_sha256'].items():
        assert sha(out / name) == digest
    assert json.loads((out / 'summary.json').read_text())['completed']
    protocol = frozen['protocol']; related = protocol['related_confirmation']
    assert sha(related['file']) == related['sha256']
    host = pd.read_csv('data/name-that-llm/train.csv', dtype={'id': str, 'prompt_id': str})
    assert sha('data/name-that-llm/train.csv') == frozen['host_sha256']
    folds = pd.read_csv(related['file'], dtype={'id': str, 'prompt_id': str, 'group': str})
    assert host.id.equals(folds.id) and host.prompt_id.equals(folds.prompt_id)
    assert folds.groupby('group').fold.nunique().max() == 1
    labels = np.asarray(frozen['labels']); fold = related['fold']; heldout = folds.fold == fold
    scores, audits = {}, []
    for arm in frozen['arms']:
        completion = json.loads((out / (arm+'-complete.json')).read_text())
        assert completion['completed'] and [r['fold'] for r in completion['folds']] == [fold]
        scores[arm], audit = audit_fold(out, remote, arm, fold, frozen, split_kind='related')
        audits.append(audit)
    sparse = sparse_reference(root, host, heldout, labels)
    lookup = dict(zip(labels, range(len(labels))))
    truth = host.loc[heldout, 'model'].map(lookup).to_numpy()
    result, predictions = score_related(truth, sparse, scores['8b_ce'], scores['4b_small_supcon'], protocol)
    result.update(verified=True, primary_gain=primary['gain'], artifact_audits=audits,
                  protocol_sha256=frozen['protocol_sha256'], limitations=protocol['limitations'])
    np.savez_compressed(root / 'related-predictions.npz', ids=host.loc[heldout, 'id'].to_numpy(dtype=str),
                        labels=labels.astype(str), candidate=predictions['candidate'], comparison=predictions['comparison'])
    (root / 'evaluation-audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
