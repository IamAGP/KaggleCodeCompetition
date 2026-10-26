import sys
from pathlib import Path
import unittest
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_v34 import balanced_accuracy, paired_interval, score_primary, save_audited_outputs, sha


class ConfirmationMetricsTest(unittest.TestCase):
    def test_reaudit_preserves_bytes_and_rejects_changed_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            arrays = {'ids': np.array(['a', 'b']), 'scores': np.eye(2)}
            save_audited_outputs(root, arrays, {'verified': True})
            before = {p.name: (sha(p), p.stat().st_mtime_ns) for p in root.iterdir()}
            save_audited_outputs(root, arrays, {'verified': True})
            self.assertEqual(before, {p.name: (sha(p), p.stat().st_mtime_ns) for p in root.iterdir()})
            with self.assertRaises(AssertionError):
                save_audited_outputs(root, dict(arrays, scores=np.zeros((2, 2))), {'verified': True})

    def test_balanced_accuracy_does_not_reward_majority_only(self):
        truth=np.array([0]*9+[1]);prediction=np.zeros(10,dtype=int)
        self.assertEqual(balanced_accuracy(truth,prediction,2),.5)
        self.assertEqual(float((truth==prediction).mean()),.9)

    def test_paired_bootstrap_preserves_exact_gain(self):
        truth=np.tile([0,0,1],20)
        groups=np.repeat(np.arange(20),3);folds=np.repeat(np.arange(5),12)
        spec={'seed':3401,'samples':1000,'interval':[.025,.975]}
        self.assertEqual(paired_interval(truth,truth,1-truth,groups,folds,spec,2),[1.,1.])
        self.assertEqual(paired_interval(truth,truth,truth,groups,folds,spec,2),[0.,0.])

    def test_bootstrap_rejects_prompt_crossing_folds(self):
        truth=np.array([0,1,0,1]);groups=np.zeros(4);folds=np.array([0,0,1,1])
        with self.assertRaises(AssertionError):
            paired_interval(truth,truth,truth,groups,folds,{'seed':1,'samples':10,'interval':[.025,.975]},2)

    def test_full_gate_uses_all_folds_and_direction(self):
        labels=np.array(['a','b']);truth=np.tile([0,1],10)
        host=pd.DataFrame({'model':labels[truth],'prompt_id':np.repeat(np.arange(10),2)})
        folds=np.repeat(np.arange(5),4);perfect=np.eye(2)[truth]*10;uninformative=np.zeros_like(perfect)
        protocol={'scoring':{'component_weight':.15},'bootstrap':{'seed':3401,'samples':1000,'interval':[.025,.975]},
                  'primary_gate':{'material_tolerance':1e-8,'candidate_pooled_balanced_accuracy_minimum':.52,
                    'candidate_minus_comparison_minimum':.02,'materially_positive_gain_folds_minimum':4}}
        passed,_=score_primary(host,folds,labels,uninformative,perfect,uninformative,protocol)
        self.assertTrue(passed['qualifies_for_related_confirmation'])
        self.assertEqual(passed['materially_positive_folds'],5)
        failed,_=score_primary(host,folds,labels,uninformative,uninformative,perfect,protocol)
        self.assertFalse(failed['qualifies_for_related_confirmation'])
        self.assertLess(failed['gain'],0)

    def test_nan_scores_are_not_scored(self):
        labels=np.array(['a','b']);host=pd.DataFrame({'model':['a','b'],'prompt_id':['p','p']})
        bad=np.array([[float('nan'),0],[0,1]])
        with self.assertRaises(AssertionError):
            score_primary(host,np.array([0,0]),labels,np.zeros((2,2)),bad,bad,{})


if __name__=='__main__':
    unittest.main()
