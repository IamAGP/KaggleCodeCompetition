"""Report numerical drift against actual saved GPU outputs, never claim score parity."""
import argparse,json
from pathlib import Path
import numpy as np


def compare(reference,candidate):
    with np.load(reference,allow_pickle=False) as a,np.load(candidate,allow_pickle=False) as b:
        assert np.array_equal(a['ids'],b['ids']) and np.array_equal(a['labels'],b['labels'])
        results={}
        for key in ['incumbent_scores','component_scores','combined_scores']:
            x,y=a[key],b[key]
            assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
            delta=np.abs(x-y)
            results[key]={'max_absolute_delta':float(delta.max()),'mean_absolute_delta':float(delta.mean()),
                          'argmax_agreement':float((x.argmax(1)==y.argmax(1)).mean())}
        return {'rows':len(a['ids']),'metrics':results,'scope':'Visible-row numerical diagnostic; not hidden-score equivalence or runtime guarantee'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cpu_scores',type=Path);p.add_argument('--reference',type=Path,default=Path('artifacts/improvement-v21/inference-output/v21_scores.npz'))
    args=p.parse_args();print(json.dumps(compare(args.reference,args.cpu_scores),indent=2))
