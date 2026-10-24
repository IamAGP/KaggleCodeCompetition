"""Append a validated frozen-feature discriminant blend to the scored v10 notebook."""
import hashlib
import json
from pathlib import Path

import nbformat


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    plan_path = Path('configs/name-that-llm/v13.json')
    plan = json.loads(plan_path.read_text())
    root = Path('artifacts/improvement-v13')
    assert sha(plan_path) == sha(root / 'frozen-plan.json')
    reports = [json.loads((root / mode / 'result.json').read_text())
               for mode in ('development', 'robustness')]
    for mode, report in zip(('development', 'robustness'), reports):
        assert report['passes'] and report['plan_sha256'] == sha(plan_path)
        assert report['source_sha256']['runner'] == sha('scripts/run_qwen_v13.py')
        for rec in json.loads((root / mode / 'artifact-manifest.json').read_text())['files']:
            p = root / mode / rec['path']
            assert sha(p) == rec['sha256'] and p.stat().st_size == rec['size_bytes']
    assert reports[0]['selected'] == reports[1]['selected']
    for path, digest in plan['inputs'].items():
        assert sha(path) == digest, path
    selected = reports[0]['selected']
    shrinkage = plan['candidates'][selected]
    source = Path('notebooks/improvement-v10/name-that-llm-v7-qwen-blend-v10.ipynb')
    nb = nbformat.read(source, as_version=4)
    nb.cells[0] = nbformat.v4.new_markdown_cell(
        '# Name That LLM — Qwen discriminant blend V13\n\n'
        'Scored v10 pipeline plus a fixed shrinkage linear discriminant probe on '
        'frozen Qwen layers 9 and 24. Training-only standardization; independent test rows. '
        'Development and related-prompt checks reuse previously inspected labels. '
        'Internet disabled. No assumed prompt groups or class counts at inference.')
    sources = {name: Path('src/name_that_llm', name).read_text()
               for name in ('qwen_probe_v11.py', 'qwen_probe_v12.py')}
    nb.cells.append(nbformat.v4.new_code_cell(
        'from pathlib import Path\nimport json\n'
        f'probe_sources = json.loads({json.dumps(sources)!r})\n'
        'for name, source in probe_sources.items():\n'
        "    (Path('name_that_llm') / name).write_text(source)\n"
        'from name_that_llm.qwen_probe_v12 import QwenDiscriminantProbe\n'
        'from name_that_llm.qwen_probe_v11 import row_z as safe_row_z\n'
        'from threadpoolctl import threadpool_limits\n'
        "assert members == ['L24_C0.02', 'L9_C0.02', 'L24_C0.2'], members\n"
        'with threadpool_limits(limits=2):\n'
        f'    discriminant = QwenDiscriminantProbe({shrinkage!r}).fit(Xtr, y)\n'
        '    discriminant_scores = discriminant.decision_function(Xte)\n'
        'final_scores = .5 * safe_row_z(blend) + .5 * safe_row_z(discriminant_scores)\n'
        "submission = pd.DataFrame({'id': test['id'].to_numpy(), 'model': np.asarray(LABELS)[final_scores.argmax(1)]})\n"
        'validate_submission(test, submission)\n'
        "submission.to_parquet('submission.parquet', index=False)\n"
        "validate_submission(test, pd.read_parquet('submission.parquet'))\n"
        f"audit = {{'experiment': 'v13', 'selected': {selected!r}, 'plan_sha256': {sha(plan_path)!r}, "
        "'rows': len(test), 'changed_from_v10': int((final_scores.argmax(1) != blend.argmax(1)).sum())}\n"
        "Path('v13_audit.json').write_text(json.dumps(audit, indent=2))\n"
        'print(json.dumps(audit), flush=True)\n'))
    nb.metadata['experiment'] = {'name': 'improvement-v13', 'v10_sha256': sha(source),
                                 'plan_sha256': sha(plan_path), 'selected': selected}
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
            compile(cell.source, '<v13>', 'exec')
    nbformat.validate(nb)
    folder = Path('notebooks/improvement-v13')
    folder.mkdir(exist_ok=True)
    name = 'name-that-llm-qwen-discriminant-v13.ipynb'
    nbformat.write(nb, folder / name)
    meta = json.loads((source.parent / 'kernel-metadata.json').read_text())
    meta.update(id='adithyagiri/name-that-llm-qwen-discriminant-v13',
                title='Name That LLM Qwen Discriminant V13', code_file=name)
    (folder / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(folder / name)


if __name__ == '__main__':
    build()
