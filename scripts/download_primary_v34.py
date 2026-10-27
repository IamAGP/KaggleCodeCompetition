"""Download the approved original V34 output; no signed URL is retained."""
import hashlib,json,sys,time,urllib.request,zipfile,subprocess
from pathlib import Path

def main():
    request=json.load(sys.stdin);archive=Path('primary-output.download.zip');last=time.monotonic();size=0
    print(json.dumps({'event':'download_primary_started','utc_epoch':time.time()}),flush=True)
    try:
        try:
            with urllib.request.urlopen(request['url'],timeout=120) as response,archive.open('wb') as out:
                while chunk:=response.read(8*1024**2):
                    out.write(chunk);size+=len(chunk)
                    if time.monotonic()-last>=10:
                        print(json.dumps({'event':'download_primary_progress','bytes':size,'utc_epoch':time.time()}),flush=True);last=time.monotonic()
        except Exception:raise RuntimeError('Primary download failed; URL suppressed') from None
        destination=Path('artifacts/improvement-v34/kaggle-output');destination.mkdir(parents=True,exist_ok=False)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                target=(destination/name).resolve()
                assert target.is_relative_to(destination.resolve()),'Unsafe archive path'
            z.extractall(destination)
        subprocess.run([sys.executable,'scripts/verify_primary_download.py','--manifest','artifacts/archive-round-20261027/v34-primary-files.json'],check=True)
    finally:archive.unlink(missing_ok=True)

if __name__=='__main__':main()
