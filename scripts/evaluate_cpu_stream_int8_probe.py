"""Audit matched three-row CPU precision probe; never claim hidden-test parity."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def evaluate(root):
    fixture=json.loads((root/'streaming-fixture.json').read_text())
    assert fixture['passed'] and fixture['forward_bitwise_equal'] and fixture['after_original_lora_merge_equal']
    runtime=json.loads((root/'runtime.json').read_text())
    with np.load(root/'fp16_reference.npz',allow_pickle=False) as a,np.load(root/'int8_member_scores.npz',allow_pickle=False) as b:
        assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['labels'],b['labels'])
        assert a['logits'].shape==b['logits'].shape==(3,18)
        assert np.isfinite(a['logits']).all() and np.isfinite(b['logits']).all()
        delta=np.abs(a['logits']-b['logits'])
        ref_top=a['logits'].argmax(1);new_top=b['logits'].argmax(1)
        margins=np.sort(a['logits'],axis=1)[:,-1]-np.sort(a['logits'],axis=1)[:,-2]
        parity={'max_absolute_logit_delta':float(delta.max()),'mean_absolute_logit_delta':float(delta.mean()),'argmax_agreement':float((ref_top==new_top).mean()),'reference_top2_margins':margins.tolist()}
    by_phase={phase:sum(t['seconds'] for t in runtime['forward_seconds'] if t['phase']==phase) for phase in ['fp16_reference','int8_dynamic']}
    assert all(value>0 for value in by_phase.values())
    result={'fixture':fixture,'parity':parity,'forward_seconds':by_phase,'speedup':by_phase['fp16_reference']/by_phase['int8_dynamic'],'int8_seconds_per_visible_row':by_phase['int8_dynamic']/3,'single_member_exceeds_entire_9h_4000row_average_budget':by_phase['int8_dynamic']/3>8.1,'scope':'Three visible rows and one14Bmember only. No ensemble/hidden accuracy or runtime guarantee. Batching/length differ onhidden.','output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'runtime.json',root/'streaming-fixture.json',root/'fp16_reference.npz',root/'int8_member_scores.npz']}}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();print(json.dumps(evaluate(args.output),indent=2))
