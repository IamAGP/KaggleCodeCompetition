"""Small independent correctness/rejection checks; no model or network access."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('verify_external',Path(__file__).resolve().parents[1]/'scripts/verify_external_validation_nov6.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

class ExternalVerifierTests(unittest.TestCase):
    def test_known_cluster_gain(self):
        # Every cluster contains all classes: exactly 2/3 improve in every resample.
        y=np.tile(np.array(['a','b','c']),20)
        b=np.full(len(y),'a');c=y.copy();groups=np.repeat(np.arange(20).astype(str),3)
        interval,meta=v.bootstrap(y,b,c,groups)
        np.testing.assert_allclose(interval,[2/3,2/3],rtol=0,atol=1e-12)
        self.assertEqual(meta['valid'],5000)
        self.assertEqual(v.metrics(y,c)['correct'],60)
        self.assertEqual(v.metrics(y,b)['per_class_correct'],{'a':20,'b':0,'c':0})

    def test_balancing_preserves_shape_and_equal_mass(self):
        p=np.random.default_rng(19).uniform(.1,1,(544,18));p/=p.sum(1,keepdims=True)
        q=v.balance(p)
        np.testing.assert_allclose(q.sum(1),1,atol=1e-12)
        np.testing.assert_allclose(q.sum(0),544/18,atol=1e-10)
        self.assertTrue(np.isfinite(q).all())

    def test_tampered_value_rejected(self):
        with self.assertRaisesRegex(ValueError,'mismatch'):
            v.close(np.array([.15,.25]),np.array([.15,.26]),'Changed vote')
        with self.assertRaisesRegex(ValueError,'Missing'):
            v.require(False,'Missing output')

if __name__=='__main__':unittest.main()
