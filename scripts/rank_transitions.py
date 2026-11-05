"""Frozen short-text TRACE adaptation. No generation, API calls, or fitting in extraction."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summaries(ranks, entropy, surprisal):
    ranks, entropy, surprisal = map(np.asarray, (ranks, entropy, surprisal))
    assert ranks.ndim == 1 and len(ranks) == len(entropy) == len(surprisal)
    assert len(ranks) and np.isfinite(ranks).all() and (ranks >= 1).all()
    assert np.isfinite(entropy).all() and np.isfinite(surprisal).all()
    buckets = np.searchsorted([10, 100, 1000], ranks, side='left')
    marginal = []
    for ix in np.array_split(np.arange(len(ranks)), 3):
        if len(ix):
            marginal.extend(np.bincount(buckets[ix], minlength=4) / len(ix))
            marginal.extend([entropy[ix].mean(), surprisal[ix].mean(), np.log(ranks[ix]).mean(), surprisal[ix].std()])
        else:
            marginal.extend([0.] * 8)
    pairs = 4*buckets[:-1] + buckets[1:]
    transitions = np.bincount(pairs, minlength=16) / max(1, len(pairs))
    return np.r_[marginal, transitions].astype(np.float32)


def shifted_scores(logits, ids, mask):
    """logits[t] predicts ids[t+1]. Input tensors [B,T,V], [B,T], [B,T]."""
    import torch
    logits = logits[:, :-1].float()
    target = ids[:, 1:]
    valid = mask[:, 1:].bool() & mask[:, :-1].bool()
    chosen = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    rank = 1 + (logits > chosen.unsqueeze(-1)).sum(-1)
    logp = torch.log_softmax(logits, dim=-1)
    entropy = -(logp.exp() * logp).sum(-1)
    surprisal = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return [(rank[i][valid[i]].cpu().numpy(), entropy[i][valid[i]].cpu().numpy(),
             surprisal[i][valid[i]].cpu().numpy()) for i in range(len(ids))]


def extract(args, plan):
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    frame = pd.read_csv(args.input, dtype={'id': str})
    assert frame.id.is_unique and frame.text.notna().all()
    if args.limit: frame = frame.iloc[:args.limit].copy()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    (out/'protocol.json').write_bytes(Path(args.plan).read_bytes())
    start = time.monotonic()
    source = args.model_path or plan['model']
    if args.model_path:
        manifest = json.loads((Path(args.model_path)/'revision-manifest.json').read_text())
        assert manifest['model'] == plan['model'] and manifest['revision'] == plan['revision']
        assert manifest['files'], 'Model file checksums required'
        for relative, digest in manifest['files'].items():
            assert sha(Path(args.model_path)/relative) == digest, relative
    options = {'local_files_only': True} if args.model_path else {'revision': plan['revision']}
    tokenizer = AutoTokenizer.from_pretrained(source, **options)
    tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32,
                                                attn_implementation='eager', **options).eval().to(args.device)
    lengths = [len(tokenizer.encode(t, add_special_tokens=False)) for t in frame.text]
    assert min(lengths) >= 2
    # Fail closed, rather than silently discard long response content.
    assert max(lengths) <= plan['max_tokens'], ('Long row requires new protocol', max(lengths))
    feats, counts, audit = [], [], []
    with torch.inference_mode():
        for begin in range(0, len(frame), plan['batch_size']):
            batch = tokenizer(frame.text.iloc[begin:begin+plan['batch_size']].tolist(),
                              add_special_tokens=False, padding=True, return_tensors='pt').to(args.device)
            logits = model(**batch, use_cache=False).logits
            scores = shifted_scores(logits, batch['input_ids'], batch['attention_mask'])
            if begin == 0:
                # Explicit causal/padding parity: standalone prefix vs full padded sequence.
                for row in range(len(batch['input_ids'])):
                    n = int(batch['attention_mask'][row].sum())
                    for pos in sorted(set([0, min(3, n-2), n-2])):
                        prefix = batch['input_ids'][row:row+1, :pos+1]
                        serial = model(input_ids=prefix, use_cache=False).logits[0, -1].float()
                        vector = logits[row, pos].float()
                        error = float((serial-vector).abs().max().cpu())
                        assert torch.allclose(serial, vector, atol=3e-4, rtol=3e-4), error
                        audit.append({'row': row, 'position': pos, 'max_logit_error': error})
            for rank, entropy, surprise in scores:
                feats.append(summaries(rank, entropy, surprise)); counts.append(len(rank))
            elapsed = time.monotonic()-start
            progress = {'rows': len(feats), 'total': len(frame), 'seconds': elapsed}
            with (out/'progress.jsonl').open('a') as f: f.write(json.dumps(progress)+'\n')
            print(json.dumps(progress), flush=True)
            if elapsed > plan['extraction_budget_seconds']:
                raise TimeoutError('Frozen extraction budget reached; preserve log and do not extend silently')
    np.savez_compressed(out/'features.npz', ids=frame.id.to_numpy(dtype=str), features=np.stack(feats), scored_tokens=counts)
    (out/'manifest.json').write_text(json.dumps({'input_sha256':sha(args.input), 'protocol_sha256':sha(out/'protocol.json'),
        'features_sha256':sha(out/'features.npz'), 'script_sha256':sha(__file__), 'model':plan['model'],
        'revision':plan['revision'], 'local_model_path':args.model_path, 'alignment_checks':audit,
        'rows':len(frame), 'seconds':time.monotonic()-start, 'device':args.device,
        'torch':torch.__version__, 'numpy':np.__version__}, indent=2))


def evaluate(args, plan):
    import pandas as pd
    from scipy.special import softmax
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    from threadpoolctl import threadpool_limits
    host = pd.read_csv(args.input, dtype={'id':str, 'prompt_id':str})
    folds = pd.read_csv(plan['folds'], dtype={'id':str, 'prompt_id':str})
    manifest=json.loads((Path(args.features).parent/'manifest.json').read_text())
    assert manifest['input_sha256']==sha(args.input) and manifest['features_sha256']==sha(args.features)
    assert manifest['protocol_sha256']==sha(args.plan)
    feat = np.load(args.features, allow_pickle=False)
    ref = np.load(plan['reference'], allow_pickle=False)
    assert np.array_equal(host.id, folds.id) and np.array_equal(host.prompt_id, folds.prompt_id)
    assert np.array_equal(host.id, feat['ids']) and np.array_equal(host.id, ref['ids'])
    assert np.array_equal(folds.fold, ref['fold']) and host.id.is_unique
    assert host.assign(fold=folds.fold).groupby('prompt_id').fold.nunique().max() == 1
    labels = ref['labels']; assert list(labels)==sorted(host.model.unique())
    assert sorted(folds.fold.unique())==[0,1,2,3,4]
    y = np.searchsorted(labels, host.model)
    assert np.array_equal(labels[y], host.model)
    x = feat['features']; assert x.shape == (len(host),40) and np.isfinite(x).all()
    assert np.isfinite(ref[plan['reference_score_key']]).all()
    base = softmax(ref[plan['reference_score_key']], axis=1)
    allp, records = {}, []
    with threadpool_limits(limits=2):
        for name, dim in [('marginal',24),('transitions',40)]:
            p = np.full_like(base, np.nan)
            for fold in sorted(folds.fold.unique()):
                tr = folds.fold.to_numpy()!=fold; va = ~tr
                assert set(host.prompt_id[tr]).isdisjoint(host.prompt_id[va])
                fit = make_pipeline(StandardScaler(), LogisticRegression(**plan['classifier']))
                fit.fit(x[tr,:dim], y[tr]); assert np.array_equal(fit[-1].classes_,np.arange(18))
                p[va] = fit.predict_proba(x[va,:dim])
                blend = (1-plan['blend_weight'])*base[va]+plan['blend_weight']*p[va]
                score = lambda v: float(balanced_accuracy_score(y[va],v.argmax(1)))
                records.append(dict(arm=name,fold=int(fold),reference=score(base[va]),
                                    standalone=score(p[va]),blend=score(blend),gain=score(blend)-score(base[va])))
            allp[name]=p
    rng=np.random.default_rng(5105); prompts=host.prompt_id.unique(); boots=[]
    pred=(.9*base+.1*allp['transitions']).argmax(1); bp=base.argmax(1)
    # Prompt-cluster resampling retains correlated responses from each prompt.
    groups=[np.flatnonzero(host.prompt_id.to_numpy()==p) for p in prompts]
    for _ in range(plan['bootstrap_replicates']):
        ix=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        boots.append(balanced_accuracy_score(y[ix],pred[ix])-balanced_accuracy_score(y[ix],bp[ix]))
    metric=lambda z:float(balanced_accuracy_score(y,z.argmax(1)))
    gain=metric(.9*base+.1*allp['transitions'])-metric(base)
    marginal=metric(.9*base+.1*allp['marginal'])
    ci=np.quantile(boots,[.025,.975]).tolist()
    result=dict(experiment=plan['experiment'], folds=records, gain=gain, prompt_bootstrap95=ci,
                passed=bool(gain>=.005 and ci[0]>0 and all(r['gain']>0 for r in records if r['arm']=='transitions')
                            and metric(.9*base+.1*allp['transitions'])>marginal),
                scope='Previously inspected prompt-disjoint component OOF; not full ensemble or LB.',
                input_sha256={p:sha(p) for p in [args.input,args.features,args.plan,plan['folds'],plan['reference']]})
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'result.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/'oof.npz',ids=host.id.to_numpy(dtype=str),labels=labels,**allp)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['extract','evaluate'])
    p.add_argument('--plan',default='configs/name-that-llm/nov5-rank-transitions.json')
    p.add_argument('--input',default='data/name-that-llm/train.csv');p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--model-path');p.add_argument('--limit',type=int)
    p.add_argument('--features');a=p.parse_args();plan=json.loads(Path(a.plan).read_text())
    (extract if a.mode=='extract' else evaluate)(a,plan)
