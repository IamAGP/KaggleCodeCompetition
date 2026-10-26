"""Validate exact coverage and DDP loss weights for uneven final batches."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('likelihood_batches',Path(__file__).parents[1]/'scripts/likelihood_batches.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('size',[1,4,7,8,15,31,32,33,2159,18000])
def test_exact_coverage_and_global_document_mean(size):
    ranks=[list(module.epoch_windows(size,3001,0,r,2,4,4)) for r in range(2)]
    seen=[]
    # Distinct scalar per-document gradients make weighting mistakes observable.
    gradients=np.arange(size,dtype=float)**2+1
    for windows in zip(*ranks):
        assert len(windows[0])==len(windows[1])
        ids=[];gradient=0.
        for steps in windows:
            assert sum(last for _,_,last in steps)==1 and steps[-1][-1]
            for indices,weight,_ in steps:
                ids.extend(indices.tolist())
                if len(indices):gradient+=weight*gradients[indices].mean()/2
                else:assert weight==0
        assert len(ids)==len(set(ids))
        np.testing.assert_allclose(gradient,gradients[ids].mean(),rtol=1e-12)
        seen.extend(ids)
    assert sorted(seen)==list(range(size))
