"""Verify official V34 kernel output download against the audited local snapshot."""
import argparse,hashlib,json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,default=Path.cwd());parser.add_argument('--manifest',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text());count=0
    for item in manifest['files']:
        if not item['path'].startswith('artifacts/improvement-v34/kaggle-output/'):continue
        path=args.repo/item['path'];assert path.is_file() and path.stat().st_size==item['bytes'],item['path']
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
        assert h.hexdigest()==item['sha256'],item['path'];count+=1
    assert count>100
    print(json.dumps({'primary_output_files_verified':count,'matches_audited_snapshot':True}))

if __name__=='__main__':main()
