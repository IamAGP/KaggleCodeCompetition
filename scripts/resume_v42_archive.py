"""Resume the existing immutable archive after a verified transfer timeout."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
out = root / 'artifacts/improvement-v42'
target = out / 'results-snapshot.tar.gz'
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-i',
       str(Path.home()/'.ssh/id_ed25519_runpod_new'), '-p', '15567', 'root@213.173.109.7']
offset = target.stat().st_size
remote = f"""
import pathlib,hashlib,json
p=pathlib.Path('/workspace/v42/results-snapshot.tar.gz')
h=hashlib.sha256();full=hashlib.sha256();remaining={offset}
with p.open('rb') as f:
 while chunk:=f.read(8*1024*1024):
  full.update(chunk)
  if remaining>0:
   part=chunk[:remaining];h.update(part);remaining-=len(part)
assert remaining==0
print(json.dumps(dict(bytes=p.stat().st_size,sha256=full.hexdigest(),prefix_sha256=h.hexdigest(),resume_offset={offset})))
"""
r=subprocess.run(ssh+['python3 -'],input=remote,text=True,capture_output=True,check=True,timeout=120)
receipt=json.loads(r.stdout)
assert receipt['bytes']>=offset
assert hashlib.sha256(target.read_bytes()).hexdigest()==receipt['prefix_sha256']
(out/'final-archive-remote-receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(dict(prefix_verified=True, remaining_bytes=receipt['bytes']-offset)),flush=True)
command=f"python3 -c 'import sys,shutil; f=open(\"/workspace/v42/results-snapshot.tar.gz\",\"rb\"); f.seek({offset}); shutil.copyfileobj(f,sys.stdout.buffer,8*1024*1024)'"
with target.open('ab') as f:
 subprocess.run(ssh+[command],stdout=f,check=True,timeout=600)
assert target.stat().st_size==receipt['bytes']
assert hashlib.sha256(target.read_bytes()).hexdigest()==receipt['sha256']
receipt['local_full_checksum_verified']=True
(out/'results-snapshot-receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
