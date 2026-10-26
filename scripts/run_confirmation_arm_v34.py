"""Run one fixed recipe on independent host folds; each fit resumes external-only."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_likelihood_v33 import prepare


def main():
    root = Path.cwd()
    plan = json.loads((root/'plan.json').read_text())
    arm = sys.argv[1]
    spec = plan['arms'][arm]
    records = []
    for fold in plan['protocol']['primary_folds']['train']:
        folder = root/arm/f'fold-{fold}'
        assert not folder.exists(), 'Never overwrite an existing fold run'
        folder.mkdir(parents=True)
        params = dict(spec['parameters'])
        params.update(experiment='improvement-v34', pilot_fold=fold,
                      labels=plan['labels'], model_dir=spec['model_dir'],
                      checkpoint=spec['checkpoint'], checkpoint_sha256=spec['checkpoint_sha256'],
                      confirmation_split=f'primary-{fold}', confirmation_fold=fold,
                      host_sha256=plan['host_sha256'], **plan['fold_sizes'][str(fold)])
        prepare(plan['host_path'], plan['fold_by_prompt'], params, folder)
        (folder/'training-plan.json').write_text(json.dumps(params,indent=2))
        (folder/'plan.json').write_text(json.dumps(params,indent=2))
        worker = root/spec['training_worker']
        env = dict(os.environ,V18_VARIANT=spec['variant'])
        def run(path, environment):
            remaining = plan['deadline_unix']-time.time()
            if remaining <= 0:
                raise TimeoutError('Confirmation notebook wall budget exhausted')
            subprocess.run([sys.executable,'-u',str(path)],cwd=folder,env=environment,
                           check=True,timeout=remaining)
        run(worker,env)
        training = json.loads((folder/spec['variant']/'result.json').read_text())
        assert training['completed'] and training['validation_rows']==params['validation_rows']
        readout = dict(params)
        readout.update(spec['readout'])
        readout.update(host_path=plan['host_path'],upstream_path=str(folder))
        weight = folder/spec['variant']/'final/adapter/adapter_model.safetensors'
        readout['adapter_sha256'] = {spec['readout_variant']:hashlib.sha256(weight.read_bytes()).hexdigest()}
        (folder/'readout-plan.json').write_text(json.dumps(readout,indent=2))
        (folder/'plan.json').write_text(json.dumps(readout,indent=2))
        try:
            run(root/spec['readout_worker'],dict(env,V19_VARIANTS=json.dumps([spec['readout_variant']])))
        finally:
            (folder/'plan.json').write_text(json.dumps(params,indent=2))
        result = json.loads((folder/(spec['readout_variant']+'_result.json')).read_text())
        record = {'arm':arm,'fold':fold,'training':training,'readout':result}
        records.append(record)
        with (root/(arm+'-progress.jsonl')).open('a') as f:
            f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    (root/(arm+'-complete.json')).write_text(json.dumps({'completed':True,'folds':records},indent=2))


if __name__ == '__main__':
    main()
