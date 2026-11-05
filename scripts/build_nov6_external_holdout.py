"""Freeze a label-stratified external audit subset without predictions or model execution."""
import hashlib,json,unicodedata
from pathlib import Path
import pandas as pd

ROOT=Path('artifacts/nov6-external-holdout-audit')
def norm(x):return ' '.join(unicodedata.normalize('NFKC',str(x)).casefold().split())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=ROOT/'frozen-544';out.mkdir(exist_ok=False)
    pool=pd.read_parquet(ROOT/'pool15.parquet')
    ids=pd.read_csv(ROOT/'provisional-cohort.csv',dtype={'id':str,'prompt_canon':str})
    assert len(ids)==17507 and ids.id.is_unique
    rows=pool.merge(ids[['id','hash']],on='id',validate='one_to_one')
    assert len(rows)==17507 and rows.model.nunique()==17
    assert all(hashlib.sha256(norm(t).encode()).hexdigest()==h for t,h in zip(rows.text,rows.hash))
    rows['selection_hash']=rows.text.map(lambda t:hashlib.sha256(('nov6-fixed-holdout-v1'+norm(t)).encode()).hexdigest())
    selected=rows.sort_values(['model','selection_hash','id']).groupby('model',sort=True).head(32).copy()
    assert len(selected)==544 and selected.groupby('model').size().eq(32).all()
    assert selected.id.is_unique and selected.hash.is_unique
    selected[['id','text']].to_csv(out/'external_test.csv',index=False)
    selected[['id','model','prompt_canon']].rename(columns={'prompt_canon':'prompt_id'}).to_csv(out/'external_labels.csv',index=False)
    selected[['id','selection_hash','source_dataset','prompt_evidence','hash']].to_csv(out/'selection-audit.csv',index=False)
    # Reserve prompt-disjoint rows, not merely different responses to already measured prompts.
    reserve=rows[~rows.prompt_canon.isin(set(selected.prompt_canon))]
    reserve[['id','model','prompt_canon','hash']].to_csv(out/'confirmation-reserve-ids.csv',index=False)
    files={f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in out.iterdir() if f.is_file()}
    manifest={'experiment':'nov6-fixed-holdout-v1','selection':'Within each of17labels choose32 smallest SHA256(seed literal concatenated with NFKC-casefold-whitespace-normalizedtext); id tie-break. No predictions used.',
     'seed_literal':'nov6-fixed-holdout-v1','rows':544,'labels':sorted(selected.model.unique()),'label_counts':selected.model.value_counts().sort_index().to_dict(),
     'prompt_id_semantics':'OpenPool Kaggleversion15 prompt_canon, own_text source evidence; mapping recomputed inpool15, so compareoldrowsusingpool15mapping.',
     'prompts':int(selected.prompt_canon.nunique()),'missing_label':'gemini-3.7-flash','prediction_columns':18,
     'evaluation_scope':'17-class macrorecall with all18 possible predicted labels. External domain shift, publisher-attestedNov2generation afterV39fittedcheckpoints. No equalmass18Sinkhorn, promptcentering or pseudo-label fitting; not exactsubmissionrecipe validation.',
     'cohort_rows':17507,'reserve_rows_prompt_disjoint':len(reserve),'reserve_prompts':int(reserve.prompt_canon.nunique()),
     'reserve_class_counts':reserve.model.value_counts().sort_index().to_dict(),
     'files':files,'source_sha256':{str(p):sha(p) for p in [ROOT/'pool15.parquet',ROOT/'pool13.parquet',ROOT/'provisional-cohort.csv',ROOT/'report.json']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'folder':str(out),'manifest_sha256':sha(out/'manifest.json'),'rows':len(selected),'prompts':manifest['prompts'],'reserve_rows':len(reserve)},indent=2))
if __name__=='__main__':main()
