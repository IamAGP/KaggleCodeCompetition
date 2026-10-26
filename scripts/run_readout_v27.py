"""One frozen nonlinear readout test on already-audited V22 features."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


def main():
    started = time.monotonic()
    root = Path('artifacts/improvement-v27')
    plan = json.loads((root/'frozen-plan.json').read_text())
    assert plan == json.loads(Path('configs/name-that-llm/v27.json').read_text())
    feature_path = Path(plan['feature_path'])
    assert hashlib.sha256(feature_path.read_bytes()).hexdigest() == plan['feature_sha256']
    host = pd.read_csv('data/name-that-llm/train.csv', dtype={'id':str})
    upstream = Path('artifacts/improvement-v22/kaggle-output')
    fit = pd.read_parquet(upstream/'training_frame.parquet')
    valid = pd.read_parquet(upstream/'validation_frame.parquet')
    fit_ids = set(fit.loc[fit.stage=='host','id'])
    mask = host.id.isin(fit_ids).to_numpy()
    assert mask.sum()==2159 and (~mask).sum()==540
    assert host.loc[~mask,'id'].tolist()==valid.id.tolist()
    assert set(host.loc[mask,'prompt_id']).isdisjoint(host.loc[~mask,'prompt_id'])
    labels = np.sort(host.model.unique())
    features = np.load(feature_path,allow_pickle=False)
    assert features.shape==(2699,3,2560) and np.isfinite(features).all()
    scores=[]
    for j,layer in enumerate(plan['layers']):
        scaler=StandardScaler().fit(features[mask,j])
        x=scaler.transform(features[mask,j]).astype(np.float64)
        v=scaler.transform(features[~mask,j]).astype(np.float64)
        gamma=1/x.shape[1]
        train_kernel=rbf_kernel(x,gamma=gamma)
        valid_kernel=rbf_kernel(v,x,gamma=gamma)
        model=SVC(C=plan['C'],kernel='precomputed',probability=True,
                  random_state=plan['seed'],class_weight='balanced',cache_size=512)
        model.fit(train_kernel,host.loc[mask,'model'])
        assert model.fit_status_==0 and np.array_equal(model.classes_,labels)
        score=model.predict_log_proba(valid_kernel)
        assert score.shape==(540,18) and np.isfinite(score).all()
        scores.append(score)
        record=dict(layer=layer,seconds=time.monotonic()-started,
                    accuracy=float((labels[score.argmax(1)]==valid.model.to_numpy()).mean()))
        with (root/'progress.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    scores=np.mean(scores,axis=0)
    sparse=np.load('artifacts/improvement-v17/uniform_pool_pilot.npz',allow_pickle=False)
    baseline=np.load(upstream/'supcon_final_validation.npz',allow_pickle=False)
    for item in [sparse,baseline]:
        assert item['ids'].tolist()==valid.id.tolist() and np.array_equal(item['labels'],labels)
    truth=valid.model.to_numpy()
    accuracy=float((labels[scores.argmax(1)]==truth).mean())
    blended=sparse['scores']+plan['blend_weight']*scores
    blend=float((labels[blended.argmax(1)]==truth).mean())
    base_accuracy=float((labels[baseline['scores'].argmax(1)]==truth).mean())
    base_blend=float((labels[(sparse['scores']+plan['blend_weight']*baseline['scores']).argmax(1)]==truth).mean())
    np.savez_compressed(root/'validation.npz',ids=valid.id.to_numpy(dtype=str),labels=labels.astype(str),scores=scores)
    result=dict(completed=True,accuracy=accuracy,baseline_accuracy=base_accuracy,blend=blend,baseline_blend=base_blend,
                qualifies=accuracy>=plan['readout_threshold'] and blend>=plan['blend_threshold'],
                seconds=time.monotonic()-started,submission_authorized=False,limitations=plan['limitations'])
    (root/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
