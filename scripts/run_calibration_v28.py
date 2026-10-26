"""Fixed prompt-cross-fitted affine score combination; no neural retraining."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, check_grad
from scipy.special import logsumexp, softmax


class ScoreCombiner:
    def __init__(self, plan):
        self.plan = plan

    def objective(self, theta, x, y):
        m = x.shape[1]
        w, b = theta[:m], theta[m:]
        z = np.einsum('nmc,m->nc', x, w) + b
        loss = (logsumexp(z, axis=1)-z[np.arange(len(y)), y]).mean()
        lw, lb = self.plan['weight_l2'], self.plan['bias_l2']
        loss += lw*np.sum(w*w) + lb*np.sum(b*b)
        residual = softmax(z, axis=1)
        residual[np.arange(len(y)), y] -= 1
        residual /= len(y)
        grad = np.r_[np.einsum('nc,nmc->m', residual, x)+2*lw*w,
                     residual.sum(0)+2*lb*b]
        return loss, grad

    def fit(self, x, y):
        centered = x-x.mean(2, keepdims=True)
        self.scale = np.maximum(np.sqrt(np.mean(centered**2, axis=(0, 2))), 1e-12)
        x = centered/self.scale[None, :, None]
        initial = np.r_[np.ones(x.shape[1]), np.zeros(x.shape[2])]
        error = check_grad(lambda t: self.objective(t, x[:18], y[:18])[0],
                           lambda t: self.objective(t, x[:18], y[:18])[1], initial)
        assert error < 1e-5, error
        opt = minimize(self.objective, initial, args=(x, y), jac=True,
                       method='L-BFGS-B',
                       bounds=[tuple(self.plan['weight_bounds'])]*x.shape[1]+[(None, None)]*x.shape[2],
                       options={k:self.plan[k] for k in ['maxiter', 'ftol', 'gtol']})
        assert opt.success and np.isfinite(opt.x).all(), opt.message
        self.theta = opt.x
        return {'weights':opt.x[:x.shape[1]].tolist(), 'bias':opt.x[x.shape[1]:].tolist(),
                'scale':self.scale.tolist(), 'iterations':int(opt.nit), 'gradient_check_error':float(error)}

    def predict(self, x):
        x = (x-x.mean(2, keepdims=True))/self.scale[None, :, None]
        return np.einsum('nmc,m->nc', x, self.theta[:x.shape[1]])+self.theta[x.shape[1]:]


def main():
    root = Path('artifacts/improvement-v28')
    root.mkdir(exist_ok=True)
    assert not (root/'result.json').exists(), 'Frozen experiment already completed'
    plan = json.loads(Path('configs/name-that-llm/v28.json').read_text())
    plan['worker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (root/'frozen-plan.json').write_text(json.dumps(plan, indent=2))
    frame = pd.read_parquet(plan['validation_frame'])
    arrays, labels = [], None
    for record in plan['inputs'].values():
        p = Path(record['path'])
        assert hashlib.sha256(p.read_bytes()).hexdigest() == record['sha256']
        d = np.load(p, allow_pickle=False)
        assert d['ids'].tolist() == frame.id.astype(str).tolist()
        if labels is None: labels = d['labels']
        assert np.array_equal(labels, d['labels'])
        assert d['scores'].shape == (540, 18) and np.isfinite(d['scores']).all()
        arrays.append(d['scores'])
    x = np.stack(arrays, axis=1).astype(np.float64)
    y = pd.Index(labels).get_indexer(frame.model)
    assert (y >= 0).all()
    groups = frame.group.to_numpy()
    unique = np.unique(groups)
    assert len(unique) == 30 and all((groups == g).sum() == 18 for g in unique)
    rng = np.random.default_rng(plan['seed'])
    folds = np.array_split(rng.permutation(unique), plan['folds'])
    scores = np.full((len(y),len(labels)), np.nan)
    fold_ids = np.full(len(y), -1)
    records = []
    for fold, heldout in enumerate(folds):
        test = np.isin(groups, heldout)
        assert set(groups[test]).isdisjoint(groups[~test])
        model = ScoreCombiner(plan)
        record = model.fit(x[~test], y[~test])
        scores[test] = model.predict(x[test])
        fold_ids[test] = fold
        record.update(fold=fold, train_rows=int((~test).sum()), test_rows=int(test.sum()),
                      heldout_groups=heldout.tolist())
        records.append(record)
        with (root/'progress.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
    assert np.isfinite(scores).all() and (fold_ids >= 0).all()
    reference = x[:,0]+0.15*x[:,1]
    correct = scores.argmax(1) == y
    old = reference.argmax(1) == y
    delta = correct.astype(float)-old.astype(float)
    means = np.array([delta[groups == g].mean() for g in unique])
    draws = rng.integers(0,len(unique),(plan['bootstrap_samples'],len(unique)))
    interval = np.quantile(means[draws].mean(1),[.025,.975])
    result = {'accuracy':float(correct.mean()),'reference_accuracy':float(old.mean()),
              'gain':float(delta.mean()),'prompt_bootstrap_95_interval':interval.tolist(),
              'corrected':int((delta>0).sum()),'newly_wrong':int((delta<0).sum()),
              'qualifies_for_further_validation':bool(correct.mean()>=plan['threshold_accuracy'] and interval[0]>0),
              'submission_authorized':False,'folds':records,'limitations':plan['limitations']}
    np.savez_compressed(root/'validation.npz',ids=frame.id.astype(str),labels=labels,scores=scores,fold=fold_ids)
    (root/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['folds','limitations']},indent=2))


if __name__ == '__main__':
    main()
