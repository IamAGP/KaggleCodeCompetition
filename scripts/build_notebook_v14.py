"""Build the scored v10 pipeline plus a separately validated Qwen3-8B probe."""
import hashlib
import json
from pathlib import Path
import nbformat
from build_features_v14 import CODE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    root = Path('artifacts/improvement-v14')
    plan_path = Path('configs/name-that-llm/v14.json')
    plan = json.loads(plan_path.read_text())
    assert sha(plan_path) == sha(root / 'frozen-plan.json')
    freeze = json.loads((root / 'evaluation-freeze.json').read_text())
    for path, digest in {**plan['inputs'], **freeze['files']}.items():
        assert sha(path) == digest, path
    reports = []
    for mode in ('development', 'robustness'):
        report = json.loads((root / mode / 'result.json').read_text())
        assert report['passes'] and report['plan_sha256'] == sha(plan_path)
        assert report['source_sha256']['runner'] == sha('scripts/run_qwen_v14.py')
        assert report['source_sha256']['probe'] == sha('src/name_that_llm/qwen_probe_v14.py')
        for rec in json.loads((root / mode / 'artifact-manifest.json').read_text())['files']:
            p = root / mode / rec['path']
            assert sha(p) == rec['sha256'] and p.stat().st_size == rec['size_bytes']
        reports.append(report)
    assert reports[0]['selected'] == reports[1]['selected']
    selected = reports[0]['selected']
    pooling = plan['candidates'][selected]
    source = Path('notebooks/improvement-v10/name-that-llm-v7-qwen-blend-v10.ipynb')
    assert sha(source) == '1ee7b4f10a786a83d1113d1ca1b9f5c6c88e76322fea0b083024394bd9896cc1'
    nb = nbformat.read(source, as_version=4)
    nb.cells[0] = nbformat.v4.new_markdown_cell(
        '# Name That LLM — Qwen8B Blend V14\n\n'
        'Preserves the scored v10 pipeline and adds a frozen Qwen3-8B reader. '
        'A fixed linear probe on layers 9/24 supplies independent passage evidence. '
        'Internet disabled. No assumed prompt groups or test class counts. '
        'Development and related-prompt validation reuse inspected labels.')
    nb.cells.append(nbformat.v4.new_code_cell(CODE.strip()))
    probe_source = Path('src/name_that_llm/qwen_probe_v14.py').read_text()
    common_source = Path('src/name_that_llm/qwen_probe_v11.py').read_text()
    nb.cells.append(nbformat.v4.new_code_cell(
        'from pathlib import Path\n'
        f"Path('name_that_llm/qwen_probe_v14.py').write_text({probe_source!r})\n"
        f"Path('name_that_llm/qwen_probe_v11.py').write_text({common_source!r})\n"
        'from name_that_llm.qwen_probe_v14 import Qwen8BProbe\n'
        'from name_that_llm.qwen_probe_v11 import row_z as safe_row_z\n'
        'from threadpoolctl import threadpool_limits\n'
        "assert members == ['L24_C0.02', 'L9_C0.02', 'L24_C0.2'], members\n"
        f"assert report['train_csv_sha256'] == {plan['inputs']['data/name-that-llm/train.csv']!r}\n"
        'del model\ngc.collect()\ntorch.cuda.empty_cache()\n'
        "y8 = train['model'].map({label:i for i,label in enumerate(LABELS)}).to_numpy()\n"
        'with threadpool_limits(limits=2):\n'
        f'    new_probe = Qwen8BProbe({pooling!r}).fit(train_features, y8)\n'
        '    new_scores = new_probe.decision_function(test_features)\n'
        'final_scores = .5*safe_row_z(blend) + .5*safe_row_z(new_scores)\n'
        "submission = pd.DataFrame({'id': test['id'].to_numpy(), 'model': np.asarray(LABELS)[final_scores.argmax(1)]})\n"
        'validate_submission(test, submission)\n'
        "submission.to_parquet('submission.parquet', index=False)\n"
        "validate_submission(test, pd.read_parquet('submission.parquet'))\n"
        f"audit = {{'experiment':'v14','selected':{selected!r},'plan_sha256':{sha(plan_path)!r},"
        "'rows':len(test),'changed_from_v10':int((final_scores.argmax(1)!=blend.argmax(1)).sum())}\n"
        "Path('v14_audit.json').write_text(json.dumps(audit,indent=2))\n"
        'print(json.dumps(audit),flush=True)\n'))
    nb.metadata['experiment'] = {'name':'improvement-v14','selected':selected,
                                 'plan_sha256':sha(plan_path),'v10_sha256':sha(source)}
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.outputs=[];cell.execution_count=None
            compile(cell.source,'<v14>','exec')
    nbformat.validate(nb)
    folder=Path('notebooks/improvement-v14');folder.mkdir(exist_ok=True)
    name='name-that-llm-qwen8b-blend-v14.ipynb'
    nbformat.write(nb,folder/name)
    meta=json.loads((source.parent/'kernel-metadata.json').read_text())
    meta.update(id='adithyagiri/name-that-llm-qwen8b-blend-v14',
                title='Name That LLM Qwen8B Blend V14',code_file=name,
                machine_shape='NvidiaTeslaT4')
    meta['model_sources'].append(plan['model'])
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(folder/name)


if __name__=='__main__': build()
