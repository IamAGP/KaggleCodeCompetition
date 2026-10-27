import json
import ast
import copy
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import nbformat
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sampling_v35 import crossed_batches
from evaluate_v35 import audit_global
from prepare_likelihood_v33 import prepare


class CrossedSamplingTest(unittest.TestCase):
    def test_rectangle_and_missing_cell(self):
        frame=pd.DataFrame([{'id':f'{a}-{p}','label':a,'group':p,'stage':'host'}
                            for a in range(18) for p in range(10) if (a,p)!=(0,0)])
        batches=list(crossed_batches(frame,1000,1801))
        self.assertEqual(batches,list(crossed_batches(frame,1000,1801)))
        for ids in batches:
            batch=frame.loc[ids]
            self.assertEqual(len(set(ids)),4)
            self.assertEqual(batch.label.nunique(),2)
            self.assertEqual(batch.group.nunique(),2)
            self.assertEqual(batch.groupby('label').group.nunique().tolist(),[2,2])
            self.assertEqual(batch.groupby('group').label.nunique().tolist(),[2,2])

    def test_external_and_ambiguous_cells_rejected(self):
        frame=pd.DataFrame([{'label':a,'group':p,'stage':'host'} for a in range(2) for p in range(2)])
        with self.assertRaises(ValueError):next(crossed_batches(frame.assign(stage='pool'),1,1))
        with self.assertRaises(ValueError):next(crossed_batches(pd.concat([frame,frame],ignore_index=True),1,1))

    def test_real_splits_and_embedded_sources(self):
        plan=json.loads(Path('artifacts/improvement-v35/frozen-plan.json').read_text())
        for fold in plan['protocol']['primary_folds']['train']:
            params=dict(plan['fold_sizes'][str(fold)],labels=plan['labels'],host_sha256=plan['host_sha256'],
                        confirmation_fold=fold,confirmation_split=f'primary-{fold}')
            with tempfile.TemporaryDirectory() as tmp:
                prepare('data/name-that-llm/train.csv',plan['fold_by_prompt'],params,tmp)
                train=pd.read_parquet(Path(tmp)/'training_frame.parquet')
                valid=pd.read_parquet(Path(tmp)/'validation_frame.parquet')
                for ids in crossed_batches(train,540,1801):
                    self.assertTrue(set(train.loc[ids,'prompt_id']).isdisjoint(valid.prompt_id))
                    self.assertTrue(set(train.loc[ids,'id']).isdisjoint(valid.id))
        a,b=plan['arms'].values()
        self.assertEqual(a['parameters'],b['parameters'])
        self.assertEqual(a['checkpoint_sha256'],b['checkpoint_sha256'])
        self.assertEqual(a['readout'],b['readout'])
        nb=nbformat.read('notebooks/training-v35/training.ipynb',4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code':compile(cell.source,'notebook','exec')
        self.assertIn('SIGKILL',nb.cells[2].source)
        meta=json.loads(Path('notebooks/training-v35/kernel-metadata.json').read_text())
        self.assertTrue(meta['is_private']);self.assertFalse(meta['enable_internet'])

class ProvenanceTest(unittest.TestCase):
    def test_global_audit_rejects_modified_worker_and_parameters(self):
        frozen=json.loads(Path('artifacts/improvement-v35/frozen-plan.json').read_text())
        notebook=nbformat.read('notebooks/training-v35/training.ipynb',4)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for node in ast.parse(notebook.cells[2].source).body:
                if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
                    call=node.value
                    if isinstance(call.func,ast.Attribute) and call.func.attr=='write_text' and isinstance(call.func.value,ast.Call):
                        pathcall=call.func.value
                        if isinstance(pathcall.func,ast.Name) and pathcall.func.id=='Path' and isinstance(call.args[0],ast.Constant):
                            (root/ast.literal_eval(pathcall.args[0])).write_text(ast.literal_eval(call.args[0]))
            remote=copy.deepcopy(frozen)
            remote['portable_runtime']=frozen['protocol']['runtime']['portable']
            (root/'plan.json').write_text(json.dumps(remote))
            (root/'summary.json').write_text(json.dumps({'completed':True}))
            shutil.copyfile('artifacts/improvement-v35/portable/run.py',root/'run.py')
            (root/'dependency-provenance.json').write_text(json.dumps({'versions':{'torch':'2.11.0+cu128','transformers':'5.16.1','peft':'0.20.0'}}))
            (root/'pip-freeze.txt').write_text('test fixture')
            audit_global(root,frozen)
            remote['arms']['crossed']['readout']['C']=1
            (root/'plan.json').write_text(json.dumps(remote))
            with self.assertRaises(AssertionError):audit_global(root,frozen)
            remote=copy.deepcopy(frozen)
            (root/'plan.json').write_text(json.dumps(remote))
            (root/'crossed_worker.py').write_text('# changed implementation')
            with self.assertRaises(AssertionError):audit_global(root,frozen)


if __name__=='__main__':unittest.main()
