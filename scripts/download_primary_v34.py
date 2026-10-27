"""Download the approved original V34 output; no signed URL is retained."""
import hashlib,json,sys,time,urllib.request,zipfile,subprocess,tarfile
from pathlib import Path

def main():
    request=json.load(sys.stdin);archive=Path('primary-output.download.zip');last=time.monotonic();size=0;digest=hashlib.sha256()
    print(json.dumps({'event':'download_primary_started','utc_epoch':time.time()}),flush=True)
    try:
        try:
            with urllib.request.urlopen(request['url'],timeout=120) as response,archive.open('wb') as out:
                while chunk:=response.read(8*1024**2):
                    out.write(chunk);size+=len(chunk);digest.update(chunk)
                    if time.monotonic()-last>=10:
                        print(json.dumps({'event':'download_primary_progress','bytes':size,'utc_epoch':time.time()}),flush=True);last=time.monotonic()
        except Exception:raise RuntimeError('Primary download failed; URL suppressed') from None
        if request.get('sha256'):assert digest.hexdigest()==request['sha256'],'Archive checksum mismatch'
        if zipfile.is_zipfile(archive):
            destination=Path('artifacts/improvement-v34/kaggle-output');destination.mkdir(parents=True,exist_ok=False)
            with zipfile.ZipFile(archive) as z:
                for name in z.namelist():
                    target=(destination/name).resolve()
                    assert target.is_relative_to(destination.resolve()),'Unsafe archive path'
                z.extractall(destination)
        else:
            assert request.get('sha256'),'Reviewed tar archive requires expected SHA256'
            with tarfile.open(archive) as t:t.extractall(Path.cwd(),filter='data')
        subprocess.run([sys.executable,'scripts/verify_primary_download.py','--manifest','artifacts/archive-round-20261027/v34-primary-files.json'],check=True)
    finally:archive.unlink(missing_ok=True)

if __name__=='__main__':main()
