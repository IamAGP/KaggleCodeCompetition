"""Copy completed V42 evidence and checkpoints from the recorded pod; verify hashes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
state = json.loads((ROOT/'artifacts/improvement-v42/deployment.json').read_text())
ssh = state['direct_ssh']
base = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-i',
        str(Path.home()/'.ssh/id_ed25519_runpod_new'), '-p', str(ssh['port']),
        f"{ssh['user']}@{ssh['host']}"]
remote = r'''
import hashlib,json,pathlib,tarfile
root=pathlib.Path('/workspace/v42')
files=[]
for name in ['dependency-lock.txt','model-input-manifest.json','plan.json','bootstrap.log','round.log','round-ready.log','v42-ready-launch.py']:
 p=root/name
 if p.exists():files.append(p)
for result in (root/'results').glob('*/fold-*/result.json'):
 if json.loads(result.read_text()).get('completed'):
  files.extend(p for p in result.parent.rglob('*') if p.is_file())
for p in (root/'results').glob('*/fold-*.log'):files.append(p)
if (root/'results/screen.json').exists():files.append(root/'results/screen.json')
manifest={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
m=root/'snapshot-manifest.json';m.write_text(json.dumps(manifest,indent=2))
archive=root/'results-snapshot.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for p in files+[m]:t.add(p,arcname=str(p.relative_to(root)))
print(json.dumps({'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'bytes':archive.stat().st_size,'files':len(files)}))
'''
r=subprocess.run(base+['python3 -'],input=remote,text=True,capture_output=True,check=True,timeout=180)
receipt=json.loads(r.stdout)
out=ROOT/'artifacts/improvement-v42';target=out/'results-snapshot.tar.gz'
with target.open('wb') as f:
 subprocess.run(base+['cat /workspace/v42/results-snapshot.tar.gz'],stdout=f,check=True,timeout=180)
assert target.stat().st_size==receipt['bytes']
assert hashlib.sha256(target.read_bytes()).hexdigest()==receipt['sha256']
(out/'results-snapshot-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
