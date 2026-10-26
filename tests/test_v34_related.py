import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import nbformat
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_related_v34 import related_package, require_primary
from prepare_likelihood_v33 import prepare
from evaluate_related_v34 import score_related


class RelatedConfirmationTest(unittest.TestCase):
    def test_related_gate_needs_candidate_floor_and_gain(self):
        truth = np.tile(np.arange(18), 30)
        perfect = np.eye(18)[truth] * 10
        neutral = np.zeros_like(perfect)
        protocol = {'scoring': {'component_weight': .15}, 'related_gate': {
            'candidate_balanced_accuracy_minimum': .5, 'candidate_minus_comparison_minimum': .015}}
        passed, _ = score_related(truth, neutral, perfect, neutral, protocol)
        self.assertTrue(passed['qualifies_for_full_host_training'])
        tied, _ = score_related(truth, neutral, perfect, perfect, protocol)
        self.assertFalse(tied['qualifies_for_full_host_training'])
        weak, _ = score_related(truth, neutral, neutral, neutral, protocol)
        self.assertFalse(weak['qualifies_for_full_host_training'])

    @classmethod
    def setUpClass(cls):
        cls.parent = json.loads(Path('artifacts/improvement-v34/frozen-plan.json').read_text())
        cls.notebook = nbformat.read('notebooks/training-v34/training.ipynb', 4)
        cls.host = pd.read_csv('data/name-that-llm/train.csv', dtype={'id': str, 'prompt_id': str})
        cls.folds = pd.read_csv(cls.parent['protocol']['related_confirmation']['file'],
                                dtype={'id': str, 'prompt_id': str, 'group': str})

    def test_missing_or_failed_primary_cannot_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(RuntimeError):
                require_primary(root)
            for audit in [{'verified': True, 'qualifies_for_related_confirmation': False},
                          {'verified': False, 'qualifies_for_related_confirmation': True}]:
                (root / 'primary-audit.json').write_text(json.dumps(audit))
                with self.assertRaises(RuntimeError):
                    require_primary(root)

    def test_related_split_preserves_models_protocol_and_group_boundaries(self):
        original = copy.deepcopy(self.parent)
        plan, notebook = related_package(self.parent, self.notebook, self.host, self.folds)
        self.assertEqual(self.parent, original)
        self.assertEqual(plan['protocol'], original['protocol'])
        self.assertEqual(plan['arms'], original['arms'])
        self.assertEqual(plan['confirmation_folds'], [0])
        self.assertEqual(notebook.cells[1].source, self.notebook.cells[1].source)
        for name, digest in original['source_sha256'].items():
            if name != 'run_confirmation_arm_v34.py':
                self.assertEqual(plan['source_sha256'][name], digest)
        params = dict(plan['fold_sizes']['0'], labels=plan['labels'], host_sha256=plan['host_sha256'],
                      confirmation_fold=0, confirmation_split='related-0')
        with tempfile.TemporaryDirectory() as temp:
            audit = prepare('data/name-that-llm/train.csv', plan['fold_by_prompt'], params, temp)
            self.assertEqual(audit['training_rows'], 2159)
            self.assertEqual(audit['validation_rows'], 540)
            self.assertEqual(audit['validation_groups'], 29)
        compile(notebook.cells[2].source, 'related-package', 'exec')

    def test_group_crossing_folds_rejected(self):
        folds = self.folds.copy()
        folds['group'] = 'one-group'
        with self.assertRaises(AssertionError):
            related_package(self.parent, self.notebook, self.host, folds)


if __name__ == '__main__':
    unittest.main()
