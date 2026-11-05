"""Synthetic output-contract tests; no model execution or score claims."""
import sys,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from scipy.special import softmax
from verify_public_control_nov5 import verify,CORE,COMP,POOL_SHA,digest


def main():
 with tempfile.TemporaryDirectory() as d:
  p=Path(d);labels=np.array([f'm{i:02}' for i in range(18)]);ids=np.array(['a','b','c'])
  def j(n,x):(p/n).write_text(json.dumps(x))
  def z(n,**x):np.savez(p/n,**x)
  b=np.random.default_rng(1).normal(size=(3,18));s=np.random.default_rng(2).normal(size=(3,18));cp=softmax(s,axis=1);c=b+.15*np.log(cp)
  pd.DataFrame({'id':ids,'text':['a','b','c']}).to_csv(p/'test.csv',index=False)
  j('manifest.json',{'labels':labels.tolist(),'weight':.15,'files':[{'path':'component_inference.py','sha256':'helper'}]})
  z('public-control-final.npz',ids=ids,labels=labels,pre_postprocessing_logits=b,final_logits=b,probabilities=softmax(b,axis=1))
  z('public-control-core.npz',ids=ids,labels=labels,**{k:b for k in CORE})
  payload=dict(ids=ids,labels=labels,baseline_logits=b,component_logits=s,component_probs=cp,combined_logits=c,probabilities=softmax(c,axis=1))
  z('public-control-ce8b.npz',**payload)
  j('public-control-completion.json',dict(required_core_members=list(CORE),completed_core_members=list(CORE),complementary_stack=True,sibling_residual=True,pairwise_verifier=True,ce8b_v39_included=False))
  j('public-control-comp-audit.json',dict(required_members=list(COMP),completed_members=list(COMP)))
  j('comp_meta.json',dict(ids=ids.tolist(),labels=labels.tolist(),family_gates_applied=True))
  np.save(p/'comp_logit.npy',b)
  j('v34_pairwise_audit.json',dict(ab_revision='03',test_rows=3,official_rows=2699,A_oof_prompt_grouped=True))
  np.save(p/'v34_pairwise_probs.npy',cp)
  j('public-control-atlas-audit.json',dict(completed=3,requested=3,budget_seconds=100))
  j('public-control-input-audit.json',dict(open_pool=dict(rows=190588,bytes=96388050,sha256_observed=POOL_SHA),packages={'transformers':'5.16.1','peft':'0.20.0'}))
  j('public-control-ce8b-audit.json',dict(component_log_weight=.15,v39_manifest_sha256=digest(p/'manifest.json'),helper_sha256='helper',rows=3,baseline_outputs_preserved=True,neural_training_at_inference=False,fusion_position='after corrected public-control final postprocessing; no extra Sinkhorn',changed_predictions=int(sum(c.argmax(1)!=b.argmax(1)))))
  for n,v in [('submission-public-control.parquet',b),('submission.parquet',c)]:pd.DataFrame({'id':ids,'model':labels[v.argmax(1)]}).to_parquet(p/n,index=False)
  assert verify(p,p/'test.csv',p/'manifest.json')['verified']
  def fails():
   try:verify(p,p/'test.csv',p/'manifest.json')
   except ValueError:return
   raise AssertionError('Corruption accepted')
  mutated=dict(payload);mutated['combined_logits']=c+.01;z('public-control-ce8b.npz',**mutated);fails()
  z('public-control-ce8b.npz',**payload)
  f=pd.read_parquet(p/'submission.parquet');f.iloc[::-1].to_parquet(p/'submission.parquet',index=False);fails();f.to_parquet(p/'submission.parquet',index=False)
  mutated=dict(payload);mutated['component_probs']=cp*.5;z('public-control-ce8b.npz',**mutated);fails()
  z('public-control-ce8b.npz',**payload)
  j('public-control-completion.json',dict(required_core_members=list(CORE),completed_core_members=list(CORE-{'deb'}),complementary_stack=True,sibling_residual=True,pairwise_verifier=True,ce8b_v39_included=False));fails()
 print('PASS: valid contract; reject corrupted log vote, shuffled IDs, unnormalized probabilities, missing member')

if __name__=='__main__':main()
