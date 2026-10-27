"""Read temporary official Kaggle download URLs from stdin; never persist URLs."""
import hashlib,json,os,shutil,sys,tarfile,urllib.request
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    root=Path(os.environ['V34_INPUT_ROOT']);partial=Path(os.environ['V34_PARTIAL_MANIFEST'])
    manifest=json.loads(partial.read_text());requests=json.load(sys.stdin)
    assert set(requests)==set(manifest['model_sources'])
    receipts=[]
    for handle,request in requests.items():
        assert handle in ['qwen-lm/qwen-3/Transformers/4b-base/1','qwen-lm/qwen-3/Transformers/8b-base/1']
        name=handle.split('/')[-2];destination=root/'models'/name
        assert not destination.exists(),'Never overwrite model snapshot'
        destination.mkdir(parents=True);temp=root/(name+'.download.tar.gz')
        try:
            try:
                with urllib.request.urlopen(request['url'],timeout=120) as response,temp.open('wb') as output:shutil.copyfileobj(response,output,8*1024*1024)
            except Exception:raise RuntimeError('Official model download failed for '+name) from None
            archive_hash=digest(temp)
            with tarfile.open(temp,'r:*') as archive:archive.extractall(destination,filter='data')
            expected=request['files'] # Root supplies exact official version-file list: name,size.
            assert isinstance(expected,list) and expected
            for item in expected:
                p=destination/item['name'];assert p.is_file() and p.stat().st_size==int(item['size']),item['name']
            for p in sorted(destination.rglob('*')):
                if p.is_file():manifest['files'].append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':digest(p)})
            receipts.append({'handle':handle,'archive_sha256':archive_hash,'files_match_official_version_listing':True,'file_count':len(expected)})
        finally:temp.unlink(missing_ok=True)
    manifest['model_snapshot_receipts_verified']=True;manifest['receipts']=receipts
    Path(os.environ['V34_INPUT_MANIFEST']).write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'model_handles':[r['handle'] for r in receipts],'complete':True}))

if __name__=='__main__':main()
