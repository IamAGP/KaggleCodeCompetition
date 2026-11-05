"""Build an offline external-cohort evaluation from verified V39 v6; never launch.

Without --cohort-dir and --dataset-ref, emits a deliberately blocked template.
Cohort manifest is the frozen research manifest.json (nov6-fixed-holdout-v1).
A separate evaluation-protocol.json preserves root decision: raw primary,
original label-free twins retained; original 18-class Sinkhorn secondary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import nbformat

SOURCE = Path('artifacts/improvement-v39/nov5-v34image-saved-notebook.ipynb')
SOURCE_SHA = '6f9931dc43c1b031658613a22726fed46f74491aea26db937149db8174f64ff2'
CANONICAL_SHA = '7bf9153d7d7ef99e02b861ba554d60f458c5cc63d8981f7f41b2bfc5572cb529'
OUT = Path('notebooks/nov6-external-validation')
TRAIN_SHA = '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

EVALUATION = r'''
# Only this final cell reads external labels. Predictions already exist on disk.
import hashlib
assert hashlib.sha256((COHORT_ROOT/'external_labels.csv').read_bytes()).hexdigest()==COHORT_FILES['external_labels.csv']['sha256']
truth=pd.read_csv(COHORT_ROOT/'external_labels.csv',dtype=str)
assert truth.columns.tolist()==['id','model','prompt_id']
assert truth.id.is_unique and not truth.isna().any().any()
assert truth.id.tolist()==test.id.tolist()
assert len(truth)==544 and truth.model.nunique()==17
assert truth.model.value_counts().eq(32).all()
assert set(truth.model)==set(LABELS)-{'gemini-3.7-flash'}
assert truth.prompt_id.str.len().gt(0).all()
# Keep all original 18 score columns; missing class is not removed or renormalized.
base_raw=softmax(V39_BASE_LOGIT)
candidate_raw=softmax(LOGIT)
base_bal=sinkhorn(base_raw.copy(),iters=300)
candidate_bal=sinkhorn(candidate_raw.copy(),iters=300)
assert np.allclose(candidate_bal,P,rtol=0,atol=1e-12)
y=truth.model.to_numpy()
classes=np.asarray(sorted(set(y)))
predictions={
 'baseline_raw':np.asarray(LABELS)[base_raw.argmax(1)],
 'candidate_raw':np.asarray(LABELS)[candidate_raw.argmax(1)],
 'baseline_balanced':np.asarray(LABELS)[base_bal.argmax(1)],
 'candidate_balanced':np.asarray(LABELS)[candidate_bal.argmax(1)],
}
def metrics(pred):
    return {'accuracy':float((pred==y).mean()),
      'balanced_accuracy_17_observed_classes':float(np.mean([(pred[y==c]==c).mean() for c in classes])),
      'correct':int((pred==y).sum()),
      'predicted_missing_gemini_3_7':int((pred=='gemini-3.7-flash').sum()),
      'per_class_correct':{c:int((pred[y==c]==c).sum()) for c in classes}}
base=predictions['baseline_raw'];candidate=predictions['candidate_raw']
groups=truth.prompt_id.to_numpy();unique=np.unique(groups)
assert len(unique)>=2,'Need at least two independent prompt clusters'
counts=np.asarray([[int(((groups==g)&(y==c)).sum()) for c in classes] for g in unique])
differences=np.asarray([[int(((groups==g)&(y==c)&(candidate==y)).sum())-int(((groups==g)&(y==c)&(base==y)).sum()) for c in classes] for g in unique])
rng=np.random.default_rng(61106);draws=[];invalid=0
for _ in range(5000):
    chosen=rng.integers(0,len(unique),len(unique));den=counts[chosen].sum(0)
    if (den==0).any():invalid+=1;continue
    draws.append(float(np.mean(differences[chosen].sum(0)/den)))
assert len(draws)>=4500,'Too many bootstrap draws omit a class; do not report unstable interval'
report={'experiment':'nov6-external-validation','manifest_sha256':COHORT_MANIFEST_SHA256,
 'rows':len(y),'observed_classes':17,'score_columns':18,'prompt_clusters':len(unique),
 'metrics':{name:metrics(pred) for name,pred in predictions.items()},
 'primary':'raw full-ensemble score argmax before Sinkhorn; original label-free centered twins retained',
 'secondary':'original 18-class Sinkhorn stress check; unrepresentative class distribution',
 'evaluation_protocol_sha256':EVALUATION_PROTOCOL_SHA256,
 'primary_delta_balanced_accuracy':metrics(candidate)['balanced_accuracy_17_observed_classes']-metrics(base)['balanced_accuracy_17_observed_classes'],
 'prompt_cluster_bootstrap_95pct':np.quantile(draws,[.025,.975]).tolist(),
 'bootstrap':{'draws':5000,'valid':len(draws),'invalid_missing_class':invalid,'seed':61106},
 'corrections':int(((candidate==y)&(base!=y)).sum()),'regressions':int(((candidate!=y)&(base==y)).sum()),
 'limitations':['External domain-shift evaluation, not host OOF or leaderboard prediction.',
 'Cohort omits Gemini 3.7; unchanged 18-class balancing intentionally retained.',
 'Public external data may differ in generation settings; selection and overlap audit must precede predictions.',
 'No weight, threshold, cluster or class-prior tuning; no automatic promotion.'],
 'submission_performed':False,'public_score_claim':None}
output=pd.DataFrame({'id':test.id.astype(str),**predictions})
output.to_parquet('external_comparison.parquet',index=False)
Path('external-validation-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
signal.alarm(0)
'''

def build(cohort_dir=None,dataset_ref=None):
    assert sha(SOURCE)==SOURCE_SHA,'Verified V39 v6 source changed'
    verified=json.loads(Path('artifacts/improvement-v39/nov5-v34image-private-run-verification.json').read_text())
    assert verified['source_cells_verified'] and verified['notebook_sha256']==CANONICAL_SHA
    canonical=Path('notebooks/improvement-v39/inference.ipynb')
    assert sha(canonical)==CANONICAL_SHA
    assert [c.source for c in nbformat.read(SOURCE,4).cells]==[c.source for c in nbformat.read(canonical,4).cells]
    digest=None
    if cohort_dir:
        assert dataset_ref,'Provide a private dataset reference with frozen cohort'
        p=Path(cohort_dir)/'manifest.json'
        m=json.loads(p.read_text())
        assert m['experiment']=='nov6-fixed-holdout-v1' and m['rows']==544
        assert len(m['labels'])==17 and set(m['label_counts'].values())=={32}
        assert m['missing_label']=='gemini-3.7-flash'
        files=m['files']
        assert {'external_test.csv','external_labels.csv'}<=set(files)
        for name in ['external_test.csv','external_labels.csv']:
            v=files[name];f=Path(cohort_dir)/name
            assert f.stat().st_size==v['bytes'] and sha(f)==v['sha256']
        digest=sha(p)
    protocol={'experiment':'nov6-external-validation','frozen_before_model_results':True,
      'cohort_manifest_sha256':digest,'rows_unchanged':True,
      'primary':'V16 versus fixed CE8B addition, raw 18-column full-ensemble score argmax before Sinkhorn; 17 observed-class balanced accuracy',
      'secondary':'Original 18-class Sinkhorn output; deliberately mismatched class-mass stress check',
      'retained':'Original label-free text clustering and centered twins, weights, models, tokenizers, training pool, numerical setup',
      'override':'Root decision supersedes research manifest recommendation against centering/Sinkhorn; cohort selection and files remain unchanged',
      'no_tuning':True,'bootstrap':{'unit':'prompt_id','draws':5000,'seed':61106},
      'claims':'External domain-shift marginal screen only; not host OOF, public LB prediction or automatic promotion'}
    OUT.mkdir(parents=True,exist_ok=True)
    protocol_path=OUT/'evaluation-protocol.json'
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    protocol_sha=sha(protocol_path)
    nb=nbformat.read(SOURCE,4)
    original_codes=[c.source for c in nb.cells if c.cell_type=='code']
    preflight=f'''import hashlib,json
from pathlib import Path
COHORT_MANIFEST_SHA256={digest!r}
EVALUATION_PROTOCOL_SHA256={protocol_sha!r}
assert COHORT_MANIFEST_SHA256 is not None,'BLOCKED TEMPLATE: frozen cohort and private dataset attachment not supplied'
roots=[p.parent for p in Path('/kaggle/input').rglob('manifest.json') if hashlib.sha256(p.read_bytes()).hexdigest()==COHORT_MANIFEST_SHA256]
assert len(roots)==1,roots
COHORT_ROOT=roots[0]
COHORT_MANIFEST=json.loads((COHORT_ROOT/'manifest.json').read_text())
assert COHORT_MANIFEST['experiment']=='nov6-fixed-holdout-v1' and COHORT_MANIFEST['rows']==544 and len(COHORT_MANIFEST['labels'])==17
COHORT_FILES=COHORT_MANIFEST['files']
assert {{'external_test.csv','external_labels.csv'}}<=set(COHORT_FILES)
p=COHORT_ROOT/'external_test.csv'
assert p.stat().st_size==COHORT_FILES['external_test.csv']['bytes']
assert hashlib.sha256(p.read_bytes()).hexdigest()==COHORT_FILES['external_test.csv']['sha256']
# Do not open external_labels.csv until final evaluation.
'''
    changed=[]
    for i,c in enumerate(nb.cells):
        if c.cell_type!='code':continue
        before=c.source
        if 'train = pd.read_csv(find("train.csv"))' in c.source:
            c.source=c.source.replace('train = pd.read_csv(find("train.csv"))',f'''train_path=find("train.csv")
assert hashlib.sha256(Path(train_path).read_bytes()).hexdigest()=={TRAIN_SHA!r}
train = pd.read_csv(train_path)''')
            c.source=c.source.replace('test = pd.read_csv(find("test.csv"))','''test = pd.read_csv(COHORT_ROOT/'external_test.csv',dtype={'id':str,'text':str},keep_default_na=False)
assert test.columns.tolist()==['id','text'] and len(test)==544 and test.id.is_unique
assert test.text.str.len().gt(0).all()''')
        if 'submission.parquet' in c.source:c.source=c.source.replace('submission.parquet','external_predictions.parquet')
        if 'v39-audit.json' in c.source:
            c.source=c.source.replace('v39-audit.json','external-runtime-audit.json').replace('signal.alarm(0)','')
        if before!=c.source:changed.append(i)
    # Model, feature extraction, member votes, tokenizer, clustering and environment remain byte-identical.
    preserved=[i for i,c in enumerate(nb.cells) if c.cell_type=='code' and i not in changed]
    nb.cells.insert(0,nbformat.v4.new_markdown_cell('# Frozen external validation of V16 versus fixed V39 CE8B\n544 external passages; 17 classes, missing Gemini 3.7. Domain-shift screen only, not host OOF or a competition submission. Read labels only after predictions. No tuning.'))
    nb.cells.insert(1,nbformat.v4.new_code_cell(preflight))
    nb.cells.append(nbformat.v4.new_code_cell(EVALUATION))
    for c in nb.cells:
        if c.cell_type=='code':compile(c.source,'external-validation','exec');c.outputs=[];c.execution_count=None
    code='\n'.join(c.source for c in nb.cells if c.cell_type=='code')
    assert 'submission.parquet' not in code
    # Preflight may name the labels file, but only final evaluation may read it.
    assert code.count("pd.read_csv(COHORT_ROOT/'external_labels.csv'")==1
    assert [nb.cells[i+2].source for i in preserved]==[nbformat.read(SOURCE,4).cells[i].source for i in preserved]
    nbformat.validate(nb);OUT.mkdir(parents=True,exist_ok=True)
    nbformat.write(nb,OUT/'external-validation.ipynb')
    meta=json.loads(Path('notebooks/improvement-v39/kernel-metadata.json').read_text())
    meta.update(id='adithyagiri/name-that-llm-external-validation-nov6',title='Name That LLM External Validation Nov6',code_file='external-validation.ipynb')
    if dataset_ref:meta['dataset_sources'].append(dataset_ref)
    (OUT/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    report={'status':'built_ready_for_review_not_launched' if digest else 'blocked_template_cohort_not_frozen',
      'source':str(SOURCE),'source_sha256':SOURCE_SHA,'canonical_source_sha256':CANONICAL_SHA,'evaluation_protocol_sha256':protocol_sha,'notebook_sha256':sha(OUT/'external-validation.ipynb'),
      'cohort_manifest_sha256':digest,'source_changed_cell_indices':changed,'preserved_code_cell_indices':preserved,
      'original_code_cells':len(original_codes),'expected_rows':544,'expected_classes':17,
      'private_external_dataset':dataset_ref,'required_missing_attachment':None if dataset_ref else 'PRIVATE frozen external cohort dataset',
      'expected_runtime_image':'gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435',
      'expected_image_pinning':'original','expected_execution':'SaveAndRunAll','expected_accelerator':'two Tesla T4 GPUs',
      'competition_train_sha256':TRAIN_SHA,'no_neural_fitting':True,'labels_read_only_in_final_cell':True,
      'no_submission_filename':True,'launch_or_upload_performed':False}
    (OUT/'build-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort-dir');p.add_argument('--dataset-ref');args=p.parse_args()
    build(args.cohort_dir,args.dataset_ref)
