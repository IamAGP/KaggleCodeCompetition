"""Describe model/provenance dependence in existing pools; no fitted text model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def describe(frame):
    table = pd.crosstab(frame.source, frame.model).sort_index().sort_index(axis=1)
    assert table.shape[1] == 18 and table.to_numpy().sum() == len(frame)
    counts = table.to_numpy(dtype=float)
    # Equal class prior: each column is P(provenance | model), then divide by 18.
    conditional = counts / counts.sum(axis=0, keepdims=True)
    joint = conditional / counts.shape[1]
    marginal = joint.sum(axis=1, keepdims=True)
    independent = marginal / counts.shape[1]
    nonzero = joint > 0
    mi = float((joint[nonzero] * np.log2((joint / independent)[nonzero])).sum())
    source_only = float(conditional.max(axis=1).sum() / counts.shape[1])
    shared = (counts > 0).all(axis=1)
    max_balanced_per_model = int(counts[shared].min(axis=1).sum())
    return {
        'rows': len(frame), 'sources': len(table), 'models': len(table.columns),
        'class_counts': frame.model.value_counts().sort_index().to_dict(),
        'source_model_counts': table.astype(int).to_dict(orient='index'),
        'equal_class_prior_source_model_mutual_information_bits': mi,
        'empirical_optimal_source_only_balanced_accuracy': source_only,
        'source_independent_chance': 1 / counts.shape[1],
        'sources_covering_every_model': table.index[shared].tolist(),
        'max_rows_without_replacement_equal_models_within_complete_sources': max_balanced_per_model * 18,
        'complete_source_minimum_model_counts': table.loc[shared].min(axis=1).astype(int).to_dict(),
        'per_model_largest_source_share': dict(zip(table.columns, conditional.max(axis=0).tolist())),
    }


def main():
    destination = Path('artifacts/diagnostics/pool-provenance-20261026.json')
    assert not destination.exists(), 'Existing diagnostic is frozen'
    paths = {'full_pool': 'data/name-that-llm/external/pool_v4.parquet',
             'v18_small_pool': 'artifacts/improvement-v18/kaggle-output/training_frame.parquet',
             'v22_large_pool': 'artifacts/improvement-v22/kaggle-output/training_frame.parquet',
             'v25_large_pool': 'artifacts/improvement-v25/kaggle-output/training_frame.parquet'}
    results, hashes = {}, {}
    for name, path in paths.items():
        frame = pd.read_parquet(path)
        if 'stage' in frame:
            frame = frame.loc[frame.stage == 'pool']
        results[name] = describe(frame)
        hashes[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert results['v22_large_pool'] == results['v25_large_pool']
    result = {'diagnostic': 'Descriptive provenance/model association, equal class prior',
              'results': results, 'input_sha256': hashes,
              'limitations': [
                  'Source-only score is an in-sample metadata calculation, not held-out text accuracy',
                  'Association does not prove that trained models exploit provenance, or explain public regression',
                  'No text classifier, training recipe, model selection or submission was changed',
                  'Source denotes pool contributor, not a verified generation-setting or topic label',
                  'Restricting to complete sources changes sample size and content; this is not a controlled performance comparison'],
              'research_context': {
                  'paper': 'https://arxiv.org/html/2104.08530v2',
                  'authors_repository': 'https://github.com/malikaltakrori/Topic-Confusion-for-authorship-attribution-EMNLP-2021-',
                  'scope': 'Historical human-authorship topic-confounding study; motivation only, not current LLM attribution SOTA or a reproduction'},
              'attribution': 'External pool: Nishant Dahal and original contributors, CC BY-SA 4.0',
              'submission_authorized': False}
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps({name: {k: v for k, v in value.items() if k not in
        ['source_model_counts', 'class_counts', 'per_model_largest_source_share']} for name, value in results.items()}, indent=2))


if __name__ == '__main__':
    main()
