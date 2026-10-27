"""Fixed training-only prompt-mean projection diagnostic; no GPU or submission."""
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from diagnose_balance_transfer import balance


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_basis(features, prompts, authors):
    """Only complete crossed training prompts contribute to content means."""
    means = []
    for prompt in np.unique(prompts):
        mask = prompts == prompt
        if mask.sum() == 18 and len(np.unique(authors[mask])) == 18:
            means.append(features[mask].mean(0, dtype=np.float64))
    means = np.stack(means)
    centre = means.mean(0)
    _, singular, vectors = svd(means-centre, full_matrices=False, check_finite=True)
    assert len(means) > 16 and singular[15] > 0
    return centre, vectors[:16], singular, len(means)


def project(features, centre, basis):
    return (features.astype(np.float64) - ((features-centre) @ basis.T) @ basis).astype(np.float32)


def metrics(truth, prediction, labels):
    matrix = confusion_matrix(truth, prediction, labels=labels)
    return dict(balanced_accuracy=float(balanced_accuracy_score(truth, prediction)),
                accuracy=float(np.mean(truth == prediction)),
                class_recall=dict(zip(labels, (matrix.diagonal()/matrix.sum(1)).tolist())),
                confusion_matrix=matrix.tolist())


def main():
    started = time.monotonic()
    config_path = Path('configs/name-that-llm/v38.json')
    config = json.loads(config_path.read_text())
    root = Path('artifacts/improvement-v38'); root.mkdir(exist_ok=False)
    (root/'protocol.json').write_text(json.dumps(config, indent=2))
    manifest = {str(config_path): sha(config_path), __file__: sha(__file__)}
    def tracked(path):
        manifest[str(path)] = sha(path)
        return Path(path)
    host_path = tracked('data/name-that-llm/train.csv')
    host = pd.read_csv(host_path, dtype={'id': str, 'prompt_id': str})
    global_plan = json.loads(tracked('artifacts/improvement-v34/frozen-plan.json').read_text())
    assert sha(host_path) == global_plan['host_sha256']
    previous = json.loads(tracked('artifacts/improvement-v34/primary-audit.json').read_text())
    assert previous['verified']
    folds_frame = pd.read_csv(tracked(global_plan['protocol']['primary_folds']['file']), dtype={'id':str,'prompt_id':str})
    assert host.id.equals(folds_frame.id) and host.prompt_id.equals(folds_frame.prompt_id)
    folds = folds_frame.fold.to_numpy()
    assert host.assign(fold=folds).groupby('prompt_id').fold.nunique().max()==1
    labels = np.asarray(global_plan['labels']); truth = host.model.to_numpy()
    sparse = np.load(tracked('artifacts/improvement-v32/oof.npz'), allow_pickle=False)
    assert np.array_equal(sparse['ids'],host.id) and np.array_equal(sparse['labels'],labels)
    assert np.array_equal(sparse['fold'],folds)
    all_scores = {rank:np.full((len(host),18),np.nan) for rank in config['ranks']}
    records=[]
    for fold in range(5):
        folder = (Path('artifacts/improvement-v25/kaggle-output') if fold==0 else
                  Path(f'artifacts/improvement-v34/kaggle-output/8b_ce/fold-{fold}'))
        feature_path = tracked(folder/'ce_final_features.npy')
        feature_result = json.loads(tracked(folder/'ce_final_result.json').read_text())
        assert sha(feature_path)==feature_result['feature_sha256']
        train_frame=pd.read_parquet(tracked(folder/'training_frame.parquet'))
        valid_frame=pd.read_parquet(tracked(folder/'validation_frame.parquet'))
        fit_ids=set(train_frame.loc[train_frame.stage=='host','id'].astype(str))
        val_ids=set(valid_frame.id.astype(str));train=folds!=fold;valid=~train
        assert fit_ids==set(host.loc[train,'id']) and val_ids==set(host.loc[valid,'id'])
        assert set(host.loc[train,'prompt_id']).isdisjoint(host.loc[valid,'prompt_id'])
        plan=json.loads(tracked(folder/'readout-plan.json').read_text())
        assert all(plan[key]==config['readout'][key] for key in ['C','max_iter','seed','layers'])
        saved=np.load(tracked(folder/'ce_final_validation.npz'),allow_pickle=False)
        assert np.array_equal(saved['ids'],host.loc[valid,'id']) and np.array_equal(saved['labels'],labels)
        features=np.load(feature_path,mmap_mode='r',allow_pickle=False)
        assert features.shape==(2699,3,4096) and np.isfinite(features).all()
        scores={rank:[] for rank in config['ranks']};projections={};diagnostics=[]
        for layer in range(3):
            centre,vectors,singular,n=prompt_basis(features[train,layer],host.loc[train,'prompt_id'].to_numpy(),truth[train])
            projections[f'centre_{layer}']=centre;projections[f'vectors_{layer}']=vectors
            diagnostics.append({'layer':config['readout']['layers'][layer],'complete_train_prompts':n,
                                'removed_prompt_variance_fraction':{str(k):float((singular[:k]**2).sum()/(singular**2).sum()) for k in config['ranks']}})
            for rank in config['ranks']:
                assert time.monotonic()-started<config['wall_limit_seconds']
                a=features[train,layer];b=features[valid,layer]
                if rank:a=project(a,centre,vectors[:rank]);b=project(b,centre,vectors[:rank])
                scaler=StandardScaler();a=scaler.fit_transform(a);b=scaler.transform(b)
                clf=LogisticRegression(C=plan['C'],max_iter=plan['max_iter'],random_state=plan['seed'])
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning);clf.fit(a,truth[train])
                assert np.array_equal(clf.classes_,labels)
                scores[rank].append(clf.predict_log_proba(b))
                print(json.dumps({'fold':fold,'layer':layer,'rank':rank,'elapsed_seconds':time.monotonic()-started}),flush=True)
        np.savez_compressed(root/f'fold-{fold}-projection.npz',**projections)
        for rank in config['ranks']:all_scores[rank][valid]=np.mean(scores[rank],axis=0)
        parity={'max_abs_score_error':float(np.max(np.abs(all_scores[0][valid]-saved['scores']))),
                'prediction_disagreements':int(np.sum(all_scores[0][valid].argmax(1)!=saved['scores'].argmax(1)))}
        row={'fold':fold,'rank0_parity':parity,'projection_diagnostics':diagnostics,'ranks':{}}
        for rank in config['ranks']:
            raw=labels[all_scores[rank][valid].argmax(1)]
            blended=labels[balance(softmax(sparse['scores'][valid]+.15*all_scores[rank][valid],axis=1)).argmax(1)]
            row['ranks'][str(rank)]={'raw':metrics(truth[valid],raw,labels),'proxy_blend':metrics(truth[valid],blended,labels)}
        records.append(row)
        (root/'partial.json').write_text(json.dumps(records,indent=2))
    pooled={}
    for rank in config['ranks']:
        assert np.isfinite(all_scores[rank]).all()
        raw=labels[all_scores[rank].argmax(1)];blended=np.empty(len(host),dtype=labels.dtype)
        for fold in range(5):
            valid=folds==fold
            blended[valid]=labels[balance(softmax(sparse['scores'][valid]+.15*all_scores[rank][valid],axis=1)).argmax(1)]
        pooled[str(rank)]={'raw':metrics(truth,raw,labels),'proxy_blend':metrics(truth,blended,labels)}
    baseline_drift=pooled['0']['proxy_blend']['balanced_accuracy']-previous['metrics']['candidate']['balanced_accuracy']
    report={'protocol_sha256':sha(config_path),'scope':'Repeated development OOF only; no submission or full-ensemble claim',
            'folds':records,'pooled':pooled,'cached_proxy_baseline_drift':baseline_drift,'seconds':time.monotonic()-started,'gpu_seconds':0,'paid_cost_usd':0}
    (root/'report.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(root/'oof.npz',ids=host.id.to_numpy(dtype=str),labels=labels,fold=folds,
                        **{f'rank_{rank}':score for rank,score in all_scores.items()})
    (root/'input-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'pooled':{k:{m:v['balanced_accuracy'] for m,v in val.items()} for k,val in pooled.items()},'seconds':report['seconds']}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
