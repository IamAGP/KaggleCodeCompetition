"""Audit the actual bounded GPU probe before allowing a long QLoRA run."""
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def main():
    root = Path('artifacts/improvement-v24')
    output = root/'kaggle-output'
    plan = json.loads((output/'plan.json').read_text())
    assert plan == json.loads(Path('configs/name-that-llm/v24.json').read_text())
    worker_hash = hashlib.sha256((output/'probe.py').read_bytes()).hexdigest()
    assert worker_hash == plan['worker_sha256']
    assert worker_hash == hashlib.sha256(Path('scripts/probe_qlora_v24.py').read_bytes()).hexdigest()
    history = [json.loads(line) for line in (output/'progress.jsonl').read_text().splitlines()]
    loaded = [row for row in history if row.get('event') == 'loaded']
    assert len(loaded) == 1 and loaded[0]['versions']['bitsandbytes'] == '0.50.0'
    modes = [row for row in history if row.get('event') == 'training_mode']
    assert len(modes) == 1 and modes[0]['encoder_training']
    assert modes[0]['checkpointed_training_modules'] >= 36
    steps = [row for row in history if row.get('event') == 'step']
    updates = [row for row in history if row.get('event') == 'update']
    assert [row['step'] for row in steps] == list(range(plan['microsteps']))
    assert [row['step'] for row in updates] == [7,15,23]
    assert all(np.isfinite(row['loss']) and row['step_seconds'] > 0 for row in steps)
    assert all(np.isfinite(row['grad_norm']) and row['skipped'] == 0 for row in updates)
    result = json.loads((output/'result.json').read_text())
    assert history[-1]['event'] == 'complete' and result['completed']
    assert result['reload_check'] == 'passed' and result['skipped_updates'] == 0
    measured = [row['step_seconds'] for row in steps[8:]]
    np.testing.assert_allclose(result['projected_5580_microsteps_seconds'], np.mean(measured)*5580)
    adapter_path = output/'adapter/adapter_model.safetensors'
    tensors = load_file(str(adapter_path))
    assert len(tensors) == 504 and all(np.isfinite(x).all() for x in tensors.values())
    assert 0 < result['changed_tensors'] <= len(tensors)
    estimate = result['projected_5580_microsteps_seconds'] + 1800
    audit = dict(verified=True, qualifies_for_v25=estimate <= 24300,
        projected_training_plus_allowance_seconds=estimate, result=result,
        adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        worker_sha256=worker_hash, versions=loaded[0]['versions'],
        limitations=['Short CE-only throughput sample; longer run can be slower',
                    'No accuracy measurement; projection includes a fixed 1800-second allowance'])
    (root/'probe-audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
