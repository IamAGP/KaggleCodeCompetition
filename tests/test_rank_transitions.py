"""Numerical tests; these do not execute the competition experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from rank_transitions import shifted_scores, summaries


def test_causal_shift_padding_and_ties():
    ids = torch.tensor([[0,2,1,3],[2,0,1,0]])
    mask = torch.tensor([[1,1,1,1],[1,1,0,0]])
    logits = torch.tensor([[[0.,2.,4.,1.],[3.,3.,1.,0.],[4.,2.,3.,0.],[0.,0.,0.,0.]],
                           [[1.,3.,4.,2.],[0.,0.,0.,0.],[9.,0.,0.,0.],[0.,0.,0.,0.]]])
    got = shifted_scores(logits, ids, mask)
    assert got[0][0].tolist()==[1,1,4]  # second token target ties for first
    assert got[1][0].tolist()==[4]      # padded target tokens excluded
    for b in range(2):
        for t in range(int(mask[b].sum())-1):
            values=logits[b,t].numpy();prob=np.exp(values-values.max());prob/=prob.sum()
            np.testing.assert_allclose(got[b][1][t],-(prob*np.log(prob)).sum(),atol=1e-6)
            np.testing.assert_allclose(got[b][2][t],-np.log(prob[ids[b,t+1]]),atol=1e-6)


def test_transition_direction_and_boundaries():
    r=np.array([1,10,11,100,101,1000,1001]);h=np.ones(7);s=np.ones(7)*2
    x=summaries(r,h,s)
    assert x.shape==(40,)
    expected=np.zeros((4,4));expected[0,0]=expected[0,1]=expected[1,1]=expected[1,2]=expected[2,2]=expected[2,3]=1/6
    np.testing.assert_allclose(x[24:].reshape(4,4),expected)
    np.testing.assert_allclose(x[:4].sum(),1)
    one=summaries([1],[0.],[0.]);assert one[24:].sum()==0 and np.isfinite(one).all()


def test_random_logits_match_serial_oracle():
    torch.manual_seed(5105)
    z=torch.randn(2,7,19);ids=torch.randint(19,(2,7));mask=torch.tensor([[1]*7,[1]*4+[0]*3])
    rows=shifted_scores(z,ids,mask)
    for b, row in enumerate(rows):
        for t, rank in enumerate(row[0]):
            assert rank==1+int((z[b,t]>z[b,t,ids[b,t+1]]).sum())
    # Future logits must not contaminate earlier scores.
    z[:,4:]=100*torch.randn_like(z[:,4:]);changed=shifted_scores(z,ids,mask)
    for b in range(2):
        for a,c in zip(rows[b],changed[b]):np.testing.assert_array_equal(a[:3],c[:3])


if __name__=='__main__':
    test_causal_shift_padding_and_ties();test_transition_direction_and_boundaries();test_random_logits_match_serial_oracle()
    print('PASS: causal target shift, padding exclusion, ties, bucket boundaries, transition orientation, serial oracle, future-logit isolation')

# A random tiny GPT2 checks the actual transformer causal path without downloading weights.
def test_transformer_prefix_parity():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(5105)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=23,bos_token_id=0,eos_token_id=0,n_positions=32,n_embd=16,n_layer=2,n_head=2,
                                   resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.)).eval()
    ids=torch.tensor([[2,8,3,9,4],[1,7,2,0,0]]);mask=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]])
    with torch.inference_mode():
        full=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
        for b in range(2):
            for pos in range(int(mask[b].sum())-1):
                serial=model(input_ids=ids[b:b+1,:pos+1],use_cache=False).logits[0,-1]
                torch.testing.assert_close(full[b,pos],serial,atol=1e-6,rtol=1e-5)
    print('PASS: actual random GPT2 full-sequence vs prefix parity, including padded batch')

if __name__=='__main__':test_transformer_prefix_parity()
