"""Package the predeclared related holdout only after a verified primary pass."""
import ast
import copy
import json
from pathlib import Path

import nbformat
import pandas as pd

from build_training_v34 import replace_once, sha


def require_primary(root):
    path = root / 'primary-audit.json'
    if not path.exists():
        raise RuntimeError('Primary V34 results are not audited; related confirmation remains unlaunched')
    audit = json.loads(path.read_text())
    if audit.get('verified') is not True or audit.get('qualifies_for_related_confirmation') is not True:
        raise RuntimeError('Primary V34 did not pass; related confirmation remains unlaunched')
    # Re-run the independent artifact audit before packaging, not just a saved pass flag.
    from evaluate_v34 import main as evaluate_primary
    evaluate_primary()
    audit = json.loads(path.read_text())
    assert audit['verified'] and audit['qualifies_for_related_confirmation']
    return audit


def related_package(parent, notebook, host, folds):
    """Pure packaging: preserve training/readout sources and the original protocol."""
    plan = copy.deepcopy(parent)
    for key in ['notebook_sha256', 'setup_sha256']:
        plan.pop(key, None)
    assert host.id.equals(folds.id) and host.prompt_id.equals(folds.prompt_id)
    assert folds.groupby('prompt_id').fold.nunique().max() == 1
    assert folds.groupby('prompt_id').group.nunique().max() == 1
    assert folds.groupby('group').fold.nunique().max() == 1
    fold = parent['protocol']['related_confirmation']['fold']
    plan['confirmation_kind'] = 'related'
    plan['confirmation_folds'] = [fold]
    plan['fold_by_prompt'] = folds.groupby('prompt_id')[['group', 'fold']].first().to_dict(orient='index')
    plan['fold_sizes'] = {str(fold): {'training_rows': int((folds.fold != fold).sum()),
                                    'validation_rows': int((folds.fold == fold).sum())}}
    source = notebook.cells[2].source
    tree = ast.parse(source)
    # Read embedded immutable worker literals rather than potentially changed local files.
    workers = {}
    plan_literal = None
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == 'write_text':
                target = call.func.value
                if isinstance(target, ast.Call) and isinstance(target.func, ast.Name) and target.func.id == 'Path':
                    if len(call.args) == 1 and isinstance(call.args[0], ast.Constant):
                        workers[ast.literal_eval(target.args[0])] = ast.literal_eval(call.args[0])
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'plan' for t in node.targets):
            plan_literal = ast.get_source_segment(source, node)
    assert set(workers) == set(parent['source_sha256']) and plan_literal
    import hashlib
    for name, text in workers.items():
        assert hashlib.sha256(text.encode()).hexdigest() == parent['source_sha256'][name]
    name = 'run_confirmation_arm_v34.py'
    worker = replace_once(workers[name], "plan['protocol']['primary_folds']['train']", "plan['confirmation_folds']")
    worker = replace_once(worker, "confirmation_split=f'primary-{fold}'", "confirmation_split=f'related-{fold}'")
    compile(worker, name, 'exec')
    source = replace_once(source, f'Path({name!r}).write_text({workers[name]!r})',
                          f'Path({name!r}).write_text({worker!r})')
    plan['source_sha256'][name] = hashlib.sha256(worker.encode()).hexdigest()
    source = replace_once(source, plan_literal, f'plan=json.loads({json.dumps(json.dumps(plan))})')
    source = replace_once(source, 'Independent multi-fold audit and frozen gates required',
                          'Independent related-holdout audit and frozen related gates required')
    compile(source, 'v34-related', 'exec')
    result = copy.deepcopy(notebook)
    result.cells[0].source = ('# V34: predeclared related-prompt confirmation\n'
        'Conditional on passing primary V34. Both fixed recipes resume external-only checkpoints. '
        'Related groups remain wholly in training or validation; no submission. '
        'Public pool and balancing: Nishant Dahal and contributors, CC BY-SA 4.0. '
        'Backbones: Qwen. LoRA: Hu et al.; SupCon: Khosla et al.; QLoRA: Dettmers et al. '
        'The original V34 protocol and attribution are embedded unchanged.')
    result.cells[2].source = source
    nbformat.validate(result)
    return plan, result


def build():
    root = Path('artifacts/improvement-v34')
    require_primary(root)
    parent = json.loads((root / 'frozen-plan.json').read_text())
    notebook_path = Path('notebooks/training-v34/training.ipynb')
    assert sha(notebook_path) == parent['notebook_sha256']
    related = parent['protocol']['related_confirmation']
    assert sha(related['file']) == related['sha256']
    host_path = Path('data/name-that-llm/train.csv')
    assert sha(host_path) == parent['host_sha256']
    host = pd.read_csv(host_path, dtype={'id': str, 'prompt_id': str})
    folds = pd.read_csv(related['file'], dtype={'id': str, 'prompt_id': str, 'group': str})
    parent['primary_audit_sha256'] = sha(root / 'primary-audit.json')
    parent['primary_oof_sha256'] = sha(root / 'primary-oof.npz')
    plan, notebook = related_package(parent, nbformat.read(notebook_path, 4), host, folds)
    folder = Path('notebooks/training-v34-related')
    destination = root / 'related'
    assert not folder.exists() and not destination.exists(), 'Never overwrite a confirmation package'
    folder.mkdir(); destination.mkdir()
    nbformat.write(notebook, folder / 'training.ipynb')
    metadata = json.loads(Path('notebooks/training-v34/kernel-metadata.json').read_text())
    metadata.update(id='adithyagiri/name-that-llm-related-confirmation-v34',
                    title='Name That LLM Related Confirmation V34')
    (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    plan['notebook_sha256'] = sha(folder / 'training.ipynb')
    (destination / 'frozen-plan.json').write_text(json.dumps(plan, indent=2))
    print(json.dumps({'folder': str(folder), 'launched': False, 'notebook_sha256': plan['notebook_sha256']}))


if __name__ == '__main__':
    build()
