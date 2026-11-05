"""Pre-fit feasibility gate only: never trains on previously inspected confirmation data."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path('artifacts/improvement-v34')
OUT = Path('artifacts/nov5-gemini-pair-feasibility')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    host_path = Path('data/name-that-llm/train.csv')
    host = pd.read_csv(host_path, dtype={'id': str})
    oof_path = ROOT / 'primary-oof.npz'
    with np.load(oof_path, allow_pickle=False) as data:
        inspected_ids = set(data['ids'].astype(str))
    assert inspected_ids == set(host.id)
    manifest = {str(p): sha(p) for p in [host_path, oof_path, Path(__file__)]}
    rows = []
    folders = [ROOT / 'kaggle-output/8b_ce' / f'fold-{i}' for i in range(1, 5)]
    folders.append(ROOT / 'related/kaggle-output/8b_ce/fold-0')
    for folder in folders:
        paths = {n: folder / n for n in ['training_frame.parquet', 'validation_frame.parquet',
            'readout-plan.json', 'data-audit.json', 'ce_final_result.json', 'ce_final_features.npy']}
        train = pd.read_parquet(paths['training_frame.parquet'])
        valid = pd.read_parquet(paths['validation_frame.parquet'])
        assert set(train.id).isdisjoint(valid.id)
        assert set(train.prompt_id).isdisjoint(valid.prompt_id)
        assert set(train.id) | set(valid.id) == set(host.id)
        plan = json.loads(paths['readout-plan.json'].read_text())
        result = json.loads(paths['ce_final_result.json'].read_text())
        audit = json.loads(paths['data-audit.json'].read_text())
        assert plan['host_sha256'] == manifest[str(host_path)]
        for n in ['training_frame.parquet', 'validation_frame.parquet']:
            assert sha(paths[n]) == audit['sha256'][n]
        features = np.load(paths['ce_final_features.npy'], mmap_mode='r', allow_pickle=False)
        assert features.shape == (len(host), 3, 4096)
        assert sha(paths['ce_final_features.npy']) == result['feature_sha256']
        assert result['fit_rows'] == len(train) and result['validation_rows'] == len(valid)
        for p in paths.values():
            manifest[str(p)] = sha(p)
        rows.append({'folder': str(folder), 'feature_shape': list(features.shape),
            'training_rows': len(train), 'validation_rows': len(valid),
            'train_validation_id_overlap': 0, 'train_validation_prompt_overlap': 0,
            'validation_rows_already_in_inspected_primary_oof': len(set(valid.id) & inspected_ids),
            'previously_unseen_validation_rows': len(set(valid.id) - inspected_ids)})
    report = {
        'experiment': 'nov5-gemini-pair-specialist-feasibility',
        'status': 'not_fitted_no_untouched_confirmation_available',
        'pair': ['gemini-3.7-flash', 'gemini-3.8-flash'],
        'feature_audit': rows,
        'generation_provenance': 'V34 evaluation checks exact adapter training frames, checkpoint identities, and readout hashes. All-row extraction uses eval/inference_mode, textual inputs only, with fold-specific trained adapters. This gate verifies stored frame/hash/shape boundaries; it does not rerun training.',
        'fixed_proposed_method_not_executed': {
            'model': 'Three binary logistic regressions, one per saved 8B layer; C=0.02, max_iter=1000, seed=1901',
            'normalization': 'StandardScaler fitted on training-pair rows only, separately per layer',
            'training': 'Only Gemini 3.7/3.8 rows from that adapter original host training split',
            'combination': 'Mean binary log probability across layers; change the winner only when the original 18-way prediction belongs to this pair; other predictions unchanged',
            'selection': 'No hyperparameter, layer, threshold, seed or blend search',
            'confirmation': None,
        },
        'reason_not_fitted': 'Pair selected after auditing all 2699 primary OOF predictions. The related holdout contains only rows from that same inspected population and was itself previously evaluated. A different fold assignment does not make those outcomes untouched.',
        'needed_to_proceed_as_confirmatory': 'A genuinely untouched labeled prompt cohort excluded from adaptation and all previous inspection, with matching frozen encoder features; otherwise an explicitly exploratory test only.',
        'public_lb_claim': None,
        'source_sha256': manifest,
    }
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'folds_audited': len(rows), 'new_confirmation_rows': sum(r['previously_unseen_validation_rows'] for r in rows)}, indent=2))

if __name__ == '__main__':
    main()
