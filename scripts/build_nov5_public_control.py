"""Build an attributed public comparison; never launch, train, or publish it."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'notebooks/nov5-public-control'


def replace_once(source, old, new):
    assert source.count(old) == 1, (old[:100], source.count(old))
    return source.replace(old, new, 1)


def build():
    import nbformat
    upstream_path = OUT / 'upstream-omer-v2.ipynb'
    original = json.loads(upstream_path.read_text())
    notebook = copy.deepcopy(original)
    for cell in notebook['cells']:
        if isinstance(cell['source'], list):
            cell['source'] = ''.join(cell['source'])
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
    cells = notebook['cells']
    cells[0]['source'] = '''# Public comparison: Omer v2, repaired and audited

This is a derived public comparison, not original model research and not a claim of our own 0.668 score. The original notebook and full contributor credits are preserved below. No V39/CE8B contribution is included yet.

Changes: fix verifier query indexing after row 127; run genuine inference on small test inputs; fail on missing required members; save aligned intermediate/final probabilities and execution evidence. These changes make this a new variant, not a byte-identical reproduction of the author's scored run. The upstream reported score has not been independently reproduced here.

Required Open Pool: Kaggle version 13 (release v11, 190,588 rows), NOT today's release v13/Kaggle version15. Input filenames, byte sizes, row counts and runtime packages are audited. Hashes are recorded on the eventual execution; they are not preverified download checksums.

Mercurydust annotation dataset license is UNKNOWN; retain its upstream terms. This draft is not a redistribution-license clearance. All other assets retain their own licenses.

''' + cells[0]['source']
    cells[2]['source'] = replace_once(cells[2]['source'],
        "FAST_COMMIT = len(_pd.read_csv(DATA + '/test.csv')) < 50   # fork: the 3-row commit run skips the heavy cells; the hidden rerun runs everything",
        "FAST_COMMIT = False  # actual inference on every input; no placeholders")
    pool_line = "_op = _pd.read_parquet(next(iter(glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True))), columns=['text', 'model', 'flag_train_prompt_regen', 'match_weight'])"
    cells[2]['source'] = replace_once(cells[2]['source'], pool_line, '''_pool_hits = glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)
assert len(_pool_hits) == 1, _pool_hits
assert os.path.getsize(_pool_hits[0]) == 96388050, 'Attach Open Pool Kaggle version 13'
_op = _pd.read_parquet(_pool_hits[0], columns=['text', 'model', 'flag_train_prompt_regen', 'match_weight'])
assert len(_op) == 190588, 'Wrong Open Pool version'
import hashlib, importlib.metadata
assert hashlib.sha256(open(_pool_hits[0], 'rb').read()).hexdigest() == '38e48b40e1db0ab1bf6e4f29d3697e37307c49d1ae5020a7af3cd59c484657d5', 'Open Pool snapshot hash mismatch'
_input_audit = {'open_pool': {'path': _pool_hits[0], 'bytes': os.path.getsize(_pool_hits[0]), 'rows': len(_op),
    'sha256_observed': hashlib.sha256(open(_pool_hits[0], 'rb').read()).hexdigest()},
    'packages': {p: importlib.metadata.version(p) for p in ['torch','transformers','peft','numpy','scipy','scikit-learn']}}
json.dump(_input_audit, open('/kaggle/working/public-control-input-audit.json', 'w'), indent=2)
import torch
assert torch.cuda.device_count() == 2 and torch.cuda.is_available(), 'Requires two actual CUDA GPUs'
assert os.environ.get('NTL_SMOKE') != '1', 'Pseudo-feature smoke mode forbidden'
''')
    # q indexes the full test array, not the current batch. Preserve all training logic.
    for old, new in [('Cte[q-start]', 'Cte[q]'), ('Wte[q-start]', 'Wte[q]'),
                     ('test_texts[z-start]', 'test_texts[z]')]:
        assert old in cells[8]['source']
        cells[8]['source'] = cells[8]['source'].replace(old, new)
    cells[7]['source'] = cells[7]['source'].replace('unchanged', 'global test-index correction')
    cells[10]['source'] = replace_once(cells[10]['source'],
        "    print('[V34.1] A/B dataset not found; proceeding without pairwise priors.', flush=True)",
        "    raise RuntimeError('Required Mercury revision03 annotations are missing')")
    main = cells[12]['source']
    main = replace_once(main, "_pairwise_path = pathlib.Path('/kaggle/working/v34_pairwise_probs.npy')", "_pairwise_path = _V38Path.Path('/kaggle/working/v34_pairwise_probs.npy')")
    main = replace_once(main, 'tot = sum(LP.values())', '''required_core = {'tfidf', 'svc4', 'distort', 'q14v5', 'q8v5', 'll8v5', 'n14b', 'q25_14', 'deb'}
assert required_core <= set(LP), (required_core - set(LP))
assert all(v.shape == (len(test), 18) and np.isfinite(v).all() for v in LP.values())
np.savez_compressed('/kaggle/working/public-control-core.npz', ids=test.id.to_numpy(dtype=str), labels=np.asarray(LABELS), **LP)
tot = sum(LP.values())''')
    main = replace_once(main, "_bundles = [d for d in a.atlas_dir.split(',') if d];", "assert a.comp_w > 0 and COMP_P is not None, 'Required complementary stack missing or invalid'\n_bundles = [d for d in a.atlas_dir.split(',') if d];")
    main = replace_once(main, "                log('ATLAS SKIPPED:', _bd, repr(_e))", "                raise RuntimeError('Required ATLAS inference failed') from _e")
    main = replace_once(main, '                completed_rows = rows[scored]', '''                completed_rows = rows[scored]
                assert len(rows) == 0 or len(completed_rows) > 0, 'ATLAS scored no requested rows'
                with open('/kaggle/working/public-control-atlas-audit.json', 'w') as fp:
                    json.dump({'requested': len(rows), 'completed': len(completed_rows), 'budget_seconds': a.atlas_budget}, fp)''')
    main = replace_once(main, "        log('V40A family-only residual failed; using exact inherited prediction:', repr(_style_error))", "        raise RuntimeError('Required sibling residual failed') from _style_error")
    main = replace_once(main, "    if _pairwise_path.is_file():", "    assert _pairwise_path.is_file(), 'Required verifier output missing'\n    if _pairwise_path.is_file():")
    main = replace_once(main, "            if _pair.shape == (_safe_logits.shape[0], 18) and np.isfinite(_pair).all():", "            assert _pair.shape == (_safe_logits.shape[0], 18) and np.isfinite(_pair).all()\n            if True:")
    main = replace_once(main, "            log('V34.1 Pairwise application error:', repr(_p_err))", "            raise RuntimeError('Required pairwise fusion failed') from _p_err")
    main = replace_once(main, "    _safe.to_parquet(_out, index=False)", '''    assert np.isfinite(_safe_logits).all()
    np.savez_compressed('/kaggle/working/public-control-final.npz', ids=test.id.to_numpy(dtype=str), labels=np.asarray(LABELS),
        pre_postprocessing_logits=_raw, final_logits=_safe_logits, probabilities=_v38sp.softmax(_safe_logits, axis=1))
    json.dump({'required_core_members': sorted(required_core), 'completed_core_members': sorted(LP),
        'complementary_stack': True, 'sibling_residual': True, 'pairwise_verifier': True,
        'ce8b_v39_included': False, 'claimed_score': None}, open('/kaggle/working/public-control-completion.json', 'w'), indent=2)
    _safe.to_parquet(_out, index=False)''')
    cells[12]['source'] = main
    comp = cells[14]['source']
    comp = replace_once(comp, "                    print('extra-passage pin skipped for', f, repr(_e), flush=True)", "                    raise RuntimeError(f'Cannot apply required extra-passage batch pin to {f}') from _e")
    comp = replace_once(comp, "        print(f'  family gate skipped {gate_key}: no fitted scores', flush=True)", "        raise RuntimeError(f'Required family gate has no fitted scores: {gate_key}')")
    comp = replace_once(comp, "        print(f'  family gate skipped {gate_key}: shape {score.shape} expected {(len(TE), len(idx))}', flush=True)", "        raise RuntimeError(f'Required family gate shape mismatch: {gate_key}: {score.shape}')")
    comp = replace_once(comp, "SMOKE = os.environ.get('NTL_SMOKE') == '1'", "SMOKE = False")
    comp = replace_once(comp, 'def add_member(key, prob, rows=None, weight=None):', 'COMPLETED_MEMBERS = []\n\ndef add_member(key, prob, rows=None, weight=None):')
    comp = replace_once(comp, '    w = W[key] if weight is None else weight', "    assert prob.ndim == 2 and prob.shape[1] == K and np.isfinite(prob).all(), key\n    COMPLETED_MEMBERS.append(key)\n    w = W[key] if weight is None else weight")
    comp = replace_once(comp, "            print(f'  SKIP {key}: checkpoint not attached ({marker})', flush=True); continue", "            if key == 'lora24_public':\n                print('DECLARED ABSENT: upstream public lora24 training asset is not attached', flush=True)\n                continue\n            raise FileNotFoundError(f'Required complementary checkpoint missing: {key}: {marker}')")
    comp = replace_once(comp, "np.save('/kaggle/working/comp_logit.npy', LOGIT.astype(np.float32))", '''required_comp = {'stylegram', 'stylometry_lgb', 'ultrabert', 'debv3l', 'electl', 'desk', 'lora24_seed7', 'lora24_claude', 'lora20_q25_instyle', 'tokattn_deb'}
assert required_comp <= set(COMPLETED_MEMBERS), required_comp - set(COMPLETED_MEMBERS)
assert np.isfinite(LOGIT).all()
json.dump({'completed_members': COMPLETED_MEMBERS, 'required_members': sorted(required_comp),
    'declared_optional_legacy_member': 'lora24_public',
    'optional_token_heads': [x[0] for x in TOKATTN_HEADS]}, open('/kaggle/working/public-control-comp-audit.json', 'w'))
np.save('/kaggle/working/comp_logit.npy', LOGIT.astype(np.float32))''')
    cells[14]['source'] = comp
    cells[16]['source'] = '''import time as _t, subprocess, sys, os
_c0 = _t.time()
os.environ['NTL_DROP_Q35'] = '0'
subprocess.run([sys.executable, 'comp_stack.py'], check=True)
COMP_SEC = _t.time() - _c0
print('comp stack wall time', round(COMP_SEC))
'''
    last = cells[18]['source']
    start = last.index('if FAST_COMMIT:')
    end = last.index('submission = pd.read_parquet', start)
    last = last[:start] + 'subprocess.run(cmd, check=True)\n' + last[end:]
    last += '''
assert submission.id.astype(str).tolist() == pd.read_csv(os.path.join(DATA, 'test.csv')).id.astype(str).tolist()
assert submission.model.isin(json.load(open('/kaggle/working/comp_meta.json'))['labels']).all()
assert os.path.isfile('/kaggle/working/public-control-completion.json')
'''
    cells[18]['source'] = last
    # Establish the audited V39 runtime before any costly public inference, for both variants.
    v39_setup = ''.join(json.loads((ROOT / 'notebooks/improvement-v39/inference.ipynb').read_text())['cells'][2]['source'])
    cells[2]['source'] = replace_once(cells[2]['source'],
        "if whl: print(subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q'] + whl, capture_output=True, text=True).returncode)",
        "# Audited bitsandbytes 0.50.0 already installed by the shared runtime cell; do not install conflicting wheels.")
    cells.insert(1, dict(nbformat.v4.new_code_cell(v39_setup)))
    notebook['nbformat_minor'] = 5
    for index, cell in enumerate(cells):
        cell['id'] = hashlib.sha256((str(index) + cell['source']).encode()).hexdigest()[:12]
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = cell['source']
            if source.startswith('%%writefile '):
                source = source.split('\n', 1)[1]
            ast.parse(source)
    notebook['metadata'].pop('kaggle', None)
    (OUT / 'inference.ipynb').write_text(json.dumps(notebook, indent=1))
    observed = json.loads((OUT / 'upstream-assets-observed.json').read_text())
    meta = observed['metadata']
    kernel = {'id': 'adithyagiri/ntl-public-control-nov5', 'title': 'NTL Public Control Nov5 - credited repaired Omer v2',
        'code_file': 'inference.ipynb', 'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': True, 'machine_shape': 'NvidiaTeslaT4', 'enable_internet': False,
        'competition_sources': meta['competition_data_sources'], 'dataset_sources': [d['ref'] + '/' + str(13 if d['ref'] == 'crtproduction/name-that-llm-open-pool' else d['observed_version']) for d in observed['datasets']],
        'kernel_sources': meta['kernel_data_sources'], 'model_sources': meta['model_data_sources']}
    kernel['dataset_sources'].append('adithyagiri/name-that-llm-offline-quantization-deps/1')
    kernel['docker_image'] = 'gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435'
    kernel['docker_image_pinning_type'] = 'original'
    (OUT / 'kernel-metadata.json').write_text(json.dumps(kernel, indent=2))
    manifest = {'status': 'local_preparation_not_executed', 'upstream': meta['ref'], 'upstream_version': 2,
        'upstream_snapshot_sha256': hashlib.sha256(upstream_path.read_bytes()).hexdigest(),
        'candidate_sha256': hashlib.sha256((OUT / 'inference.ipynb').read_bytes()).hexdigest(),
        'required_open_pool_kaggle_version': 13, 'required_open_pool_bytes': 96388050, 'required_open_pool_rows': 190588,
        'asset_observations': observed['datasets'], 'required_annotation_revision': '03',
        'unknown_license': ['mercurydust/ntllm-chatgpt-a-b'], 'ce8b_v39_included': False,
        'interface': 'public-control-final.npz: ids,labels,pre_postprocessing_logits,final_logits,probabilities; public-control-core.npz contains weighted core members',
        'remaining': ['Dataset refs explicitly versioned; confirm resolved versions by server readback',
            'Notebook-output refs need resolved version verification; observed versions are listed separately',
            'Confirm runtime package compatibility; full GPU execution has not been tested',
            'Review unknown annotation license before redistribution', 'Real visible-row execution then member audit'],
        'public_score_claim': None}
    (OUT / 'build-manifest.json').write_text(json.dumps(manifest, indent=2))
    build_ce8b(notebook, kernel, manifest)
    print(json.dumps({'built': str(OUT), 'sha256': manifest['candidate_sha256'], 'launched': False}))


def build_ce8b(public_notebook, public_metadata, public_manifest):
    """Exploratory fixed post-public-control log vote; verified helper stays unchanged."""
    import nbformat
    asset = ROOT / 'artifacts/improvement-v39/dataset'
    manifest_path = asset / 'v39-asset-manifest.json'
    m = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert m['weight'] == 0.15 and m['layers'] == [12, 24, 36]
    helper_record = next(f for f in m['files'] if f['path'] == 'component_inference.py')
    assert hashlib.sha256((asset / helper_record['path']).read_bytes()).hexdigest() == helper_record['sha256']
    v39 = json.loads((ROOT / 'notebooks/improvement-v39/inference.ipynb').read_text())
    setup = ''.join(v39['cells'][2]['source'])
    assert '5.16.1' in setup and '0.20.0' in setup
    addition = f'''# Identical verified V39 helper and weights; only surrounding fusion differs.
import hashlib, importlib.util, gc, shutil, time
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.special import softmax
V39_MANIFEST_SHA256 = {digest!r}
roots = [p.parent for p in Path('/kaggle/input').rglob('v39-asset-manifest.json') if hashlib.sha256(p.read_bytes()).hexdigest() == V39_MANIFEST_SHA256]
assert len(roots) == 1, roots
v39_root = roots[0]
v39_manifest = json.loads((v39_root/'v39-asset-manifest.json').read_text())
for item in v39_manifest['files']:
    p = v39_root/item['path']
    assert p.stat().st_size == item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], item['path']
assert v39_manifest['weight'] == 0.15 and not v39_manifest['neural_training_at_inference']
test = pd.read_csv(os.path.join(DATA, 'test.csv'))
test['text'] = test['text'].fillna('').astype(str)
baseline = np.load('/kaggle/working/public-control-final.npz', allow_pickle=False)
ids = test.id.to_numpy(dtype=str)
labels = baseline['labels'].astype(str).tolist()
assert len(set(labels)) == 18 and np.array_equal(baseline['ids'].astype(str), ids)
assert baseline['final_logits'].shape == (len(test), 18) and np.isfinite(baseline['final_logits']).all()
completion = json.load(open('/kaggle/working/public-control-completion.json'))
assert set(completion['required_core_members']) <= set(completion['completed_core_members'])
assert completion['complementary_stack'] and completion['sibling_residual'] and completion['pairwise_verifier']
shutil.copyfile('/kaggle/working/submission.parquet', '/kaggle/working/submission-public-control.parquet')
base_cfg = json.load(open(os.path.join(B8, 'config.json')))
assert base_cfg['model_type'] == 'qwen3' and base_cfg['hidden_size'] == 4096 and base_cfg['num_hidden_layers'] == 36
spec = importlib.util.spec_from_file_location('v39_component', v39_root/'component_inference.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
assert torch.cuda.device_count() == 2 and torch.cuda.is_available()
component = module.ContrastiveReadout(B8, str(v39_root/'adapter'), layers=v39_manifest['layers'], max_length=v39_manifest['max_length'], batch_size=v39_manifest['batch_size'])
features = component.features(test.text.tolist())
component_labels, scores = component.predict(features, v39_root/'readout.npz')
assert len(set(component_labels)) == 18 and set(component_labels) == set(labels)
scores = scores[:, [list(component_labels).index(label) for label in labels]]
assert scores.shape == baseline['final_logits'].shape and np.isfinite(scores).all()
component_probs = softmax(scores, axis=1)
combined_logits = baseline['final_logits'] + 0.15 * np.log(np.clip(component_probs, 1e-9, 1))
combined_probs = softmax(combined_logits, axis=1)
assert np.isfinite(combined_probs).all() and np.allclose(combined_probs.sum(1), 1)
sub = pd.DataFrame({{'id': ids, 'model': np.asarray(labels)[combined_logits.argmax(1)]}})
assert sub.id.is_unique and sub.model.isin(labels).all() and sub.id.tolist() == ids.tolist()
sub.to_parquet('/kaggle/working/submission.parquet', index=False)
assert pd.read_parquet('/kaggle/working/submission.parquet').equals(sub)
np.savez_compressed('/kaggle/working/public-control-ce8b.npz', ids=ids, labels=np.asarray(labels), baseline_logits=baseline['final_logits'], component_logits=scores, component_probs=component_probs, combined_logits=combined_logits, probabilities=combined_probs)
audit = {{'experiment': 'nov5-public-control-plus-ce8b', 'component_log_weight': 0.15,
    'fusion_position': 'after corrected public-control final postprocessing; no extra Sinkhorn',
    'v39_manifest_sha256': V39_MANIFEST_SHA256, 'helper_sha256': {helper_record['sha256']!r},
    'rows': len(test), 'changed_predictions': int(np.sum(combined_logits.argmax(1) != baseline['final_logits'].argmax(1))),
    'original_contribution': 'Our previously trained CE8B adapter/readout only; public pipeline credited to all original contributors',
    'public_score_claim': None, 'full_ensemble_validation_claim': None,
    'baseline_outputs_preserved': True, 'neural_training_at_inference': False}}
Path('/kaggle/working/public-control-ce8b-audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit), flush=True)
del component, features; gc.collect(); torch.cuda.empty_cache()
'''
    ast.parse(addition)
    candidate = copy.deepcopy(public_notebook)
    candidate['cells'].insert(0, dict(nbformat.v4.new_markdown_cell('# Exploratory: credited public control + our fixed CE8B contribution\nThe complete corrected public baseline runs first and its outputs remain saved. The unchanged verified V39 helper adds a fixed 0.15 log-probability vote AFTER public postprocessing. This differs from V39’s V16 fusion position and is not a validated ensemble gain. No baseline gains are claimed as our original work.')))
    candidate['cells'].append(dict(nbformat.v4.new_code_cell(addition)))
    candidate['nbformat_minor'] = 5
    for index, cell in enumerate(candidate['cells']):
        cell['id'] = hashlib.sha256((str(index) + cell['source']).encode()).hexdigest()[:12]
    nbformat.validate(nbformat.from_dict(candidate))
    path = OUT / 'inference-ce8b.ipynb'
    path.write_text(json.dumps(candidate, indent=1))
    metadata = copy.deepcopy(public_metadata)
    metadata.update(id='adithyagiri/ntl-public-control-ce8b-nov5', title='NTL Credited Public Control plus CE8B Nov5', code_file='inference-ce8b.ipynb',
        docker_image='gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435', docker_image_pinning_type='original')
    metadata['dataset_sources'] += ['adithyagiri/name-that-llm-v39-private-ce8b/1']
    (OUT / 'kernel-metadata-ce8b.json').write_text(json.dumps(metadata, indent=2))
    receipt = {'status': 'local_preparation_not_executed', 'baseline_candidate_sha256': public_manifest['candidate_sha256'],
        'candidate_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'v39_asset_manifest_sha256': digest,
        'helper_sha256': helper_record['sha256'], 'component_log_weight': 0.15,
        'fusion_position': 'after public-control final postprocessing', 'helper_unchanged': True,
        'known_good_v39_environment_image': metadata['docker_image'],
        'remaining': ['GPU compatibility of combined public recipe and V39 pinned environment', 'Resolved input metadata readback', 'Genuine small-test execution and runtime budget assessment'],
        'public_score_claim': None, 'baseline_outputs_preserved': True}
    (OUT / 'build-manifest-ce8b.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    build()
