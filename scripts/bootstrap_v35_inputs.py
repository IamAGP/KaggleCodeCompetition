"""Copy exact local V35 host/checkpoint assets into the shared input tree."""
import hashlib,json,os,shutil
from pathlib import Path


def main():
    root=Path(os.environ['V34_INPUT_ROOT']);root.mkdir(parents=True,exist_ok=False)
    plan=json.loads(Path('artifacts/improvement-v35/frozen-plan.json').read_text())
    origin=Path('artifacts/improvement-v18/kaggle-output/ce_supcon/external_only')
    files=[(Path('data/name-that-llm/train.csv'),Path('train.csv'),plan['host_sha256'])]
    files.extend((origin/p,Path('checkpoint/ce_supcon/external_only')/p,h) for p,h in plan['arms']['independent']['checkpoint_sha256'].items())
    entries=[]
    for source,relative,expected in files:
        assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        entries.append({'path':str(relative),'bytes':target.stat().st_size,'sha256':expected})
    manifest={'model_sources':['qwen-lm/qwen-3/Transformers/4b-base/1'],'model_snapshot_receipts_verified':False,'files':entries}
    Path(os.environ['V34_PARTIAL_MANIFEST']).write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
