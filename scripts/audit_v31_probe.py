"""Verify exact V31 sources, both GPU outcomes, adapter parity and runtime gate."""
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def main():
    root=Path('artifacts/improvement-v31');out=root/'kaggle-output-v2'
    plan=json.loads((root/'frozen-plan.json').read_text())
    remote=json.loads((out/'plan.json').read_text())
    assert {k:v for k,v in plan.items() if k!='notebook_sha256'}==remote
    for name,sha in plan['source_sha256'].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
    results=[];checkpoints=[]
    for rank in range(2):
        result=json.loads((out/f'probe-result-rank{rank}.json').read_text())
        events=[json.loads(s) for s in (out/f'progress-rank{rank}.jsonl').read_text().splitlines()]
        steps=[e for e in events if e['event']=='microstep']
        diagnostics=[e for e in events if e['event']=='gradient_diagnostic']
        assert [e['step'] for e in steps]==list(range(24))
        assert [e['step'] for e in diagnostics]==list(range(24))
        assert all(not e['nonfinite_parameters'] and e['scale']==128 for e in diagnostics)
        assert all(np.isfinite(e['loss']) and e['seconds']>0 for e in steps)
        assert [e['step'] for e in steps if e['update']]==[3,7,11,15,19,23]
        assert events[-1]['event']=='complete'
        mean=np.mean([e['seconds'] for e in steps[8:]])
        np.testing.assert_allclose(mean,result['mean_training_microstep_seconds'],rtol=1e-12)
        pilot=mean*plan['projected_training_microsteps']+result['all_source_batch_seconds']*540/4+600
        hidden=result['all_source_batch_seconds']*4500/4+3600
        np.testing.assert_allclose([pilot,hidden],[result['projected_pilot_seconds'],result['projected_hidden_seconds']],rtol=1e-12)
        assert result['verified_updates'] and result['checkpoint_reload_verified'] and result['changed_tensors']==225
        tensors=load_file(out/f'probe-adapter-rank{rank}.safetensors')
        assert len(tensors)==225 and all(np.isfinite(t).all() for t in tensors.values())
        if checkpoints:
            assert set(tensors)==set(checkpoints[0])
            for name,value in tensors.items():np.testing.assert_array_equal(value,checkpoints[0][name])
        checkpoints.append(tensors);results.append(result)
    result={'verified':True,'source_sha256':plan['source_sha256'],'replicas_identical':True,
            'qualifies_for_pilot':all(r['qualifies_for_pilot'] and
                max(r['projected_pilot_seconds'],r['projected_hidden_seconds'])<=plan['launch_gate_seconds'] for r in results),
            'rank_results':results,'checkpoint_discard_required':True}
    (root/'probe-audit-v2.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
