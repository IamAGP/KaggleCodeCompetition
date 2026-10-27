"""Run prepared fixed protocols; parent owns pod billing, transfer and termination."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path


def descendants(pid):
    children={}
    for p in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=p.read_text().rsplit(')',1)[1].split(); children.setdefault(int(fields[1]),[]).append(int(p.parent.name))
        except (OSError,ValueError,IndexError):continue
    result=[];queue=[pid]
    while queue:
        current=queue.pop();result.append(current);queue.extend(children.get(current,[]))
    return result[::-1]


def process_start(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError):return None


def terminate_tree(pid,tracked=None):
    # Retain children observed before a failed runner exited and orphaned them.
    targets=dict(tracked or {})
    for target in descendants(pid):targets.setdefault(target,process_start(target))
    for sig in [signal.SIGTERM,signal.SIGKILL]:
        for target,started in targets.items():
            if started is None or process_start(target)!=started:continue
            try:os.kill(target,sig)
            except ProcessLookupError:pass
        if sig==signal.SIGTERM:time.sleep(2)


def run(command,cwd,env,deadline,logpath):
    remaining=deadline-time.time()
    if remaining<=0:raise TimeoutError('Round deadline exhausted')
    with logpath.open('w') as log:
        process=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        tracked={};next_heartbeat=0
        try:
            while process.poll() is None:
                for pid in descendants(process.pid):tracked.setdefault(pid,process_start(pid))
                if time.time()>=deadline:raise TimeoutError('Stage deadline exhausted')
                if time.time()>=next_heartbeat:
                    log.flush()
                    logpath.with_suffix('.heartbeat.json').write_text(json.dumps({'updated_unix':time.time(),'pid':process.pid,'log_bytes':logpath.stat().st_size,'seconds_remaining':deadline-time.time()}))
                    next_heartbeat=time.time()+10
                time.sleep(min(.5,max(.01,deadline-time.time())))
            if process.returncode:raise RuntimeError('Stage failed; inspect '+str(logpath))
        finally:
            if process.poll() is None or process.returncode:
                terminate_tree(process.pid,tracked);process.wait()


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--input-manifest',type=Path,required=True);p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--allow-v35',action='store_true');p.add_argument('--mode',choices=['v34','v35'],required=True);a=p.parse_args()
    root=a.repo.resolve(); logs=root/'artifacts/runpod-round-logs';logs.mkdir(exist_ok=False)
    env=dict(os.environ,V34_INPUT_ROOT=str(a.input_root.resolve()),V35_INPUT_ROOT=str(a.input_root.resolve()),V34_INPUT_MANIFEST=str(a.input_manifest.resolve()),CUDA_VISIBLE_DEVICES='0')
    def interrupt(*_):raise TimeoutError('Round interrupted')
    signal.signal(signal.SIGTERM,interrupt)
    deadline=a.deadline_unix-300 # Leave five minutes for parent transfer/termination.
    events=[]
    def stage(name,script,cwd,cap):
        begin=time.time();record={'stage':name,'started_unix':begin,'status':'running'};events.append(record)
        (logs/'state.json').write_text(json.dumps(events,indent=2))
        try:run([sys.executable,'-u',str(script)],cwd,env,min(deadline,begin+cap),logs/(name+'.log'))
        except BaseException:
            record.update(status='failed',finished_unix=time.time());raise
        else:record.update(status='complete',finished_unix=time.time())
        finally:(logs/'state.json').write_text(json.dumps(events,indent=2))
    def run_v35():
        if deadline-time.time()<9300:raise TimeoutError('Insufficient reserved V35 time')
        v35=root/'artifacts/improvement-v35';out=v35/'runpod-output';out.mkdir(exist_ok=False)
        import shutil
        shutil.copyfile(v35/'portable/run.py',out/'run.py')
        stage('v35-setup',v35/'portable/setup.py',root,300)
        # Verify every provided asset, including exact official model download receipt.
        import hashlib
        manifest=json.loads(a.input_manifest.read_text())
        assert manifest['model_snapshot_receipts_verified'] is True
        assert 'qwen-lm/qwen-3/Transformers/4b-base/1' in manifest['model_sources']
        for entry in manifest['files']:
            path=a.input_root/entry['path'];assert path.stat().st_size==entry['bytes']
            h=hashlib.sha256()
            with path.open('rb') as source:
                for block in iter(lambda:source.read(8*1024*1024),b''):h.update(block)
            assert h.hexdigest()==entry['sha256'],entry['path']
        stage('v35-training',out/'run.py',out,9000)
        run([sys.executable,'-u',str(root/'scripts/evaluate_v35.py'),str(out)],root,env,min(deadline,time.time()+300),logs/'v35-audit.log')
    if a.mode=='v35':
        run_v35()
        (logs/'finished.json').write_text(json.dumps({'complete':True,'mode':'v35','pod_termination_required':True,'events':events},indent=2))
        return
    related=root/'artifacts/improvement-v34/related'
    assert not (related/'kaggle-output').exists(),'Never rerun completed/partial experiment silently'
    stage('setup',related/'portable/setup.py',root,300)
    (related/'kaggle-output').mkdir()
    stage('related-training',related/'portable/run.py',related/'kaggle-output',9000)
    stage('related-audit',root/'scripts/evaluate_related_v34.py',root,1800)
    audit=json.loads((related/'evaluation-audit.json').read_text())
    assert audit['verified']
    if audit['qualifies_for_full_host_training']:
        if deadline-time.time()<9300:raise TimeoutError('Related passed; insufficient reserved time for fullhost; preserve result')
        stage('full-build',root/'scripts/build_full_v34.py',root,300)
        full=root/'artifacts/improvement-v34/full-host';(full/'kaggle-output').mkdir()
        stage('full-training',full/'run.py',full/'kaggle-output',9000)
        stage('full-audit',root/'scripts/audit_full_v34.py',root,300)
    if a.allow_v35 and deadline-time.time()>=9300:
        run_v35()
    (logs/'finished.json').write_text(json.dumps({'complete':True,'submission_created':False,'pod_termination_required':True,'events':events},indent=2))

if __name__=='__main__':main()
