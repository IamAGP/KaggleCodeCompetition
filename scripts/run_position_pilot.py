"""New, fixed CPU pilot of sentence-position views; not a historical rerun."""
from __future__ import annotations
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import signal
import time
import warnings

# Bound resource use while other pilots share this computer.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/position-pilot-20261026'
SEED = 261026


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parts(text):
    # Heuristic splitter, deliberately fixed; abbreviations can over-split.
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', str(text).strip()) if s.strip()]
    if len(sentences) == 1:
        return sentences[0], '', ''
    if not sentences:
        return '', '', ''
    return sentences[0], ' '.join(sentences[1:-1]), sentences[-1]


def vectorize(fit_text, val_text):
    blocks = []
    feature_counts = []
    for analyzer, ngram, cap in [('char', (3, 5), 60000), ('word', (1, 2), 20000)]:
        v = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=2,
                            max_features=cap, sublinear_tf=True, lowercase=False,
                            dtype=np.float32)
        blocks.append((v.fit_transform(fit_text), v.transform(val_text)))
        feature_counts.append(len(v.vocabulary_))
    return (normalize(hstack([b[0] for b in blocks], format='csr')),
            normalize(hstack([b[1] for b in blocks], format='csr')), feature_counts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'result.json').exists() or (OUT/'protocol.json').exists():
        raise RuntimeError('Existing experiment found; no overwrite or silent rerun.')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(OUT/'run.log'), logging.StreamHandler()])
    started = time.perf_counter()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('15 minute CPU pilot limit')))
    signal.alarm(900)
    train_path = ROOT/'data/name-that-llm/train.csv'
    folds_path = ROOT/'artifacts/baseline-v1/folds.csv'
    df = pd.read_csv(train_path, dtype={'id': str, 'prompt_id': str})
    folds = pd.read_csv(folds_path, dtype={'id': str, 'prompt_id': str})
    assert df.id.equals(folds.id) and df.prompt_id.equals(folds.prompt_id)
    fit = folds.fold.ne(0).to_numpy()
    val = ~fit
    assert set(df.loc[fit,'prompt_id']).isdisjoint(df.loc[val,'prompt_id'])
    protocol = {
        'name': 'position-pilot-20261026', 'frozen_before_fit': True,
        'created_utc': pd.Timestamp.now(tz='UTC').isoformat(),
        'question': 'Do separate opening/middle/ending text views improve a new host-only whole-text reference?',
        'data': {'train_sha256': sha(train_path), 'folds_sha256': sha(folds_path),
                 'fit_rows': int(fit.sum()), 'val_rows': int(val.sum()), 'held_out_fold': 0},
        'source_sha256': sha(__file__), 'sklearn': sklearn.__version__,
        'split': 'First sentence, middle sentences, last sentence. One sentence goes only into opening; other regions empty. Two sentences have empty middle.',
        'features_per_view': {'char': [3,5], 'char_cap':60000, 'word':[1,2], 'word_cap':20000,
                              'min_df':2, 'lowercase':False, 'sublinear_tf':True, 'normalization':'L2 each view, then L2 combined'},
        'arms': ['whole', 'whole_plus_positions'], 'classifier': {'type':'LinearSVC','C':1.0,'max_iter':10000,'seed':SEED},
        'decision_rule': 'Worth independent confirmation only if gain >= 0.01 balanced accuracy and 95% paired prompt-bootstrap gain interval excludes zero.',
        'bootstrap': {'unit':'prompt','replicates':2000,'seed':SEED},
        'budget': {'max_seconds':900,'gpu':False,'paid_api_calls':0,'external_rows':0},
        'limitations': ['New pilot reference is not the full incumbent ensemble.',
                       'Fold zero has been inspected historically; this is not an untouched holdout.',
                       'Candidate increases feature capacity as well as adding position; gain would not isolate causality.',
                       'Sentence splitter is heuristic.', 'Related prior V14 prefix-128 frozen Qwen features failed; this is a different sparse positional test.']
    }
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    logging.info('Protocol frozen; train=%d validation=%d', fit.sum(),val.sum())
    allparts = np.array([parts(t) for t in df.text], dtype=object)
    logging.info('Build whole-text view')
    wfit, wval, counts = vectorize(df.loc[fit,'text'],df.loc[val,'text'])
    matrices = {'whole': (wfit,wval)}
    feature_counts = {'whole': counts}
    pfit,pval=[wfit],[wval]
    for i,name in enumerate(['opening','middle','ending']):
        logging.info('Build %s view', name)
        a,b,c = vectorize(allparts[fit,i],allparts[val,i])
        pfit.append(a);pval.append(b);feature_counts[name]=c
    matrices['whole_plus_positions']=(normalize(hstack(pfit,format='csr')),normalize(hstack(pval,format='csr')))
    predictions={}; scores={}; timings={}; iterations={}; warning_log=[]
    for name,(a,b) in matrices.items():
        arm_start=time.perf_counter()
        logging.info('Fit %s with %d columns',name,a.shape[1])
        clf=LinearSVC(C=1.0,dual=True,max_iter=10000,random_state=SEED)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            clf.fit(a,df.loc[fit,'model'])
            warning_log.extend(str(w.message) for w in ws)
        predictions[name]=clf.predict(b)
        scores[name]=clf.decision_function(b)
        iterations[name]=int(clf.n_iter_)
        timings[name]=time.perf_counter()-arm_start
        logging.info('%s balanced_accuracy=%.6f seconds=%.3f',name,balanced_accuracy_score(df.loc[val,'model'],predictions[name]),timings[name])
    truth=df.loc[val,'model'].to_numpy(); groups=df.loc[val,'prompt_id'].to_numpy()
    old=predictions['whole']==truth;new=predictions['whole_plus_positions']==truth
    unique=np.unique(groups);rng=np.random.default_rng(SEED)
    indices={p:np.flatnonzero(groups==p) for p in unique}
    diffs=[]
    for _ in range(2000):
        ix=np.concatenate([indices[p] for p in rng.choice(unique,len(unique),replace=True)])
        diffs.append(balanced_accuracy_score(truth[ix],predictions['whole_plus_positions'][ix])-balanced_accuracy_score(truth[ix],predictions['whole'][ix]))
    interval=np.quantile(diffs,[.025,.975]).tolist()
    metrics={k:float(balanced_accuracy_score(truth,v)) for k,v in predictions.items()}
    gain=metrics['whole_plus_positions']-metrics['whole']
    result={'metrics':metrics,'gain':gain,'paired_prompt_bootstrap_95pct':interval,
            'corrected_errors':int((~old&new).sum()),'new_errors':int((old&~new).sum()),
            'both_wrong':int((~old&~new).sum()),'both_correct':int((old&new).sum()),
            'feature_counts':feature_counts,'fit_predict_seconds':timings,'solver_iterations':iterations,
            'warnings':warning_log,'runtime_seconds':time.perf_counter()-started,
            'one_sentence_rows':int(np.sum((allparts[:,1]=='')&(allparts[:,2]==''))),
            'decision':'confirm' if gain>=.01 and interval[0]>0 else 'do_not_promote',
            'protocol_sha256':sha(OUT/'protocol.json'),'limitations':protocol['limitations']}
    np.savez_compressed(OUT/'validation_predictions.npz',ids=df.loc[val,'id'].to_numpy(dtype=str),
                        prompt_ids=groups.astype(str),truth=truth.astype(str),labels=clf.classes_.astype(str),
                        whole_predictions=predictions['whole'].astype(str),candidate_predictions=predictions['whole_plus_positions'].astype(str),
                        whole_scores=scores['whole'],candidate_scores=scores['whole_plus_positions'])
    result['prediction_sha256']=sha(OUT/'validation_predictions.npz')
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    logging.info('RESULT %s',json.dumps(result))
    signal.alarm(0)

if __name__=='__main__': main()
