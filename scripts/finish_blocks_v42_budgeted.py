"""Finish V42 under a recorded allocation within the user's unchanged $10 cap.

Execution-only recovery: original plan, trainer, inputs and completed fits remain
unchanged. Uses the explicitly recorded revised deadline and 300-second cleanup reserve.
A 660-second start reserve exceeds 1.5 times the observed 427–435-second
end-to-end intervals between completed fits (including process startup).
Run only after confirming the original runner has exited; never concurrently."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--budget-addendum", required=True)
    for key in ['plan','input-root','model-dir','output','control-receipt']:p.add_argument('--'+key,required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text())
    receipt=json.loads(Path(a.control_receipt).read_text())
    assert receipt['watchdog_verified'] is True
    assert receipt['opening_balance_usd'] >= receipt['max_new_spend_usd']
    assert receipt['deadline_unix'] > time.time()
    addendum=json.loads(Path(a.budget_addendum).read_text())
    assert addendum['v42_plan_sha256']==hashlib.sha256(Path(a.plan).read_bytes()).hexdigest()
    assert receipt['pod_id']==addendum['pod_id']
    assert receipt['deadline_unix']==addendum['revised_deadline_unix']
    assert receipt['max_new_spend_usd']==addendum['revised_round_cap_usd']
    assert addendum['prior_posted_usd']+addendum['delayed_billing_reserve_usd']+receipt['max_new_spend_usd'] <= addendum['user_cap_usd']
    assert addendum['user_cap_usd']==10
    assert addendum['revised_deadline_unix']-addendum['original_deadline_unix']==900
    assert receipt['max_new_spend_usd']==1.75
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    code=Path(__file__).resolve().parent
    for name,digest in plan['code_sha256'].items():
        assert hashlib.sha256((code/name).read_bytes()).hexdigest()==digest,name
    deadline=min(time.time()+plan['round_training_seconds'],receipt['deadline_unix']-300)
    for fold in plan['folds']:
        for arm in plan['arms']:
            folder=root/arm/f'fold-{fold}'
            # A finished fit may be reused only with matching recorded configuration.
            if (folder/'result.json').exists():
                result=json.loads((folder/'result.json').read_text())
                config=json.loads((folder/'run-config.json').read_text())
                assert result['completed'] and config==dict(plan=plan,arm=arm,fold=fold)
                continue
            if folder.exists():raise RuntimeError(f'Partial fit must be diagnosed before retry: {folder}')
            remaining=deadline-time.time()
            if remaining < min(660, plan['per_fit_seconds']):raise TimeoutError('Insufficient reserved time for next fit')
            command=[sys.executable,'-u',str(code/'train_blocks_v42.py'),'--plan',str(Path(a.plan).resolve()),
                '--input-root',str(Path(a.input_root).resolve()),'--model-dir',a.model_dir,'--output',str(folder),
                '--arm',arm,'--fold',str(fold)]
            folder.parent.mkdir(parents=True,exist_ok=True)
            with (folder.parent/f'fold-{fold}.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                    timeout=min(remaining,plan['per_fit_seconds']+60),env=dict(os.environ,TOKENIZERS_PARALLELISM='false'))
            print(json.dumps({'completed_arm':arm,'fold':fold,'time':time.time()}),flush=True)
    subprocess.run([sys.executable,str(code/'evaluate_blocks_v42.py'),'--plan',a.plan,'--output',str(root),
                    '--reference-root',str(Path(a.input_root)/'v40-reference')],check=True)

if __name__=='__main__':main()
