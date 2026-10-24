"""Reproduce Nishant Dahal's public two-stage notebook with unchanged inference cells."""
from pathlib import Path
import hashlib
import json
import nbformat


AUDIT = '''from pathlib import Path
assert torch.cuda.device_count() >= 2, 'Dual GPUs required'
assert len(train)==2699 and train.id.is_unique and test.id.is_unique
assert set(train.model.unique()) == set(LABELS) and not pd.isna(y).any()
MEMBER_AUDIT=[]
_original_vote=vote
def vote(name,P,w,rows=slice(None)):
    assert P.ndim==2 and P.shape[1]==K and np.isfinite(P).all()
    assert np.allclose(P.sum(1),1,atol=1e-5)
    MEMBER_AUDIT.append({'member':name,'weight':float(w),'rows':len(P)})
    return _original_vote(name,P,w,rows)
print({'audit':'real inference on every input','devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},flush=True)
'''


def build():
    root=Path('artifacts/improvement-v16')
    original=root/'upstream-source.ipynb'
    digest=hashlib.sha256(original.read_bytes()).hexdigest()
    nb=nbformat.read(original,as_version=4)
    original_code=[c.source for c in nb.cells if c.cell_type=='code']
    nb.nbformat_minor=5
    nb.cells.insert(0,nbformat.v4.new_markdown_cell(
        '# Name That LLM — Two-stage public reproduction V16\n\n'
        'Reproduction of [Nishant Dahal, Two-stage LoRA, part two, version 2]'
        '(https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two). '
        'Upstream documentation and all credits follow unchanged. The reported 0.620 public '
        'score and 0.642 CV belong to the upstream author, not this run. We have not '
        'independently reproduced its training CV. Our previous best is 0.427.\n\n'
        'Public pretrained assets: [NTL handwriting blend]'
        '(https://www.kaggle.com/datasets/nishantdahal/ntl-handwriting-blend), version 1, '
        'CC BY-SA 4.0. No private synthetic data added. Original blend weights, centered '
        'heads and Sinkhorn balancing are retained, including their test-distribution '
        'assumptions. Added execution audits and output validation only. All inputs '
        'perform actual model inference.'))
    nb.cells.insert(3,nbformat.v4.new_code_cell(AUDIT))
    nb.cells.append(nbformat.v4.new_code_cell(f'''
assert set(m['member'] for m in MEMBER_AUDIT) >= {{'svc','lora_qwen3_14b','lora_qwen25_14b','deberta'}}
assert np.isfinite(LOGIT).all() and np.isfinite(P).all()
assert sub.columns.tolist()==['id','model'] and sub.id.tolist()==test.id.astype(str).tolist()
assert sub.id.is_unique and sub.model.isin(LABELS).all()
assert pd.read_parquet('submission.parquet').equals(sub)
audit={{'experiment':'v16','upstream':'nishantdahal/two-stage-lora-part-two','upstream_version':2,'upstream_source_sha256':{digest!r},'rows':len(test),'members':MEMBER_AUDIT,'elapsed_seconds':time.time()-T0,'claimed_own_cv':None}}
Path('v16_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit),flush=True)
'''))
    assert [c.source for c in nb.cells if c.cell_type=='code' and c.source in original_code]==original_code
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    for c in nb.cells:
        if c.cell_type=='code':
            c.outputs=[];c.execution_count=None
            compile(c.source,'<v16>','exec')
    nbformat.validate(nb)
    folder=Path('notebooks/improvement-v16');folder.mkdir(exist_ok=True)
    name='name-that-llm-two-stage-v16.ipynb'
    nbformat.write(nb,folder/name)
    meta={'id':'adithyagiri/name-that-llm-two-stage-v16','title':'Name That LLM Two Stage V16',
          'code_file':name,'language':'python','kernel_type':'notebook','is_private':True,
          'enable_gpu':True,'machine_shape':'NvidiaTeslaT4','enable_internet':False,'enable_tpu':False,
          'competition_sources':['name-that-llm'],'dataset_sources':['nishantdahal/ntl-handwriting-blend'],
          'model_sources':['qwen-lm/qwen-3/Transformers/14b-base/1','qwen-lm/qwen2.5/Transformers/14b/1'],
          'kernel_sources':[]}
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    plan={'experiment':'improvement-v16','type':'credited public recipe reproduction',
          'upstream':'https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two','version':2,
          'source_sha256':digest,'notebook_sha256':hashlib.sha256((folder/name).read_bytes()).hexdigest(),
          'reported_upstream_score':.620,'our_incumbent_score':.427,
          'release_policy':'Submit once real visible-row inference completes with all four required members, valid ordered IDs/schema/labels and finite probabilities. No new local CV claimed.',
          'changes':['Attribution header','Dual GPU and label checks','Member audits without modifying predictions','Parquet and input ID validation'],
          'limitations':['Upstream CV not independently reproduced; pretrained heads saw host training labels.','Original centered twins and Sinkhorn assume a particular test distribution.','No gain claimed before our scored run.'],
          'asset_license':'CC BY-SA 4.0, original credits retained',
          'sources':['https://arxiv.org/html/2106.09685v2','https://github.com/microsoft/LoRA','https://arxiv.org/html/2111.09543v4','https://github.com/microsoft/DeBERTa']}
    s=json.dumps(plan,indent=2)+'\n'
    Path('configs/name-that-llm/v16.json').write_text(s)
    (root/'frozen-plan.json').write_text(s)
    print('Validated unchanged upstream inference cells:',len(original_code))
    print('Frozen notebook:',plan['notebook_sha256'])


if __name__=='__main__': build()
