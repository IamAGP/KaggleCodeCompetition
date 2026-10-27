import ast
import json
import hashlib
import nbformat
import os
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_notebook_v39 import inference_source,build
from quicksave_v39 import QuickSaveRequest,cell_sources,normalize_metadata


class InferencePackageTest(unittest.TestCase):
    def test_sdk_camelcase_metadata_readback(self):
        actual=normalize_metadata({'datasetDataSources':['owner/private'],
            'modelDataSources':['owner/model/Transformers/base/1'],
            'competitionDataSources':['name-that-llm'],'isPrivate':True,
            'enableGpu':True,'enableInternet':False,'enableTpu':False})
        self.assertEqual(actual['dataset_data_sources'],['owner/private'])
        self.assertEqual(actual['model_data_sources'],['owner/model/Transformers/base/1'])
        self.assertEqual(actual['competition_data_sources'],['name-that-llm'])
        self.assertTrue(actual['is_private'] and actual['enable_gpu'])
        self.assertFalse(actual['enable_internet'] or actual['enable_tpu'])
        self.assertEqual(actual.get('kernel_data_sources') or [],[])

    def test_inference_extraction_preserves_methods_removes_fitting(self):
        source=Path('scripts/component_v26.py').read_text()
        extracted=inference_source(source)
        before=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef))
        after=next(n for n in ast.parse(extracted).body if isinstance(n,ast.ClassDef))
        self.assertEqual([n.name for n in after.body],['__init__','features','predict'])
        for method in after.body:
            original=next(n for n in before.body if isinstance(n,ast.FunctionDef) and n.name==method.name)
            self.assertEqual(ast.dump(method),ast.dump(original))
        self.assertFalse(any(isinstance(n,ast.FunctionDef) and n.name=='main' for n in ast.parse(extracted).body))
        self.assertNotIn('LogisticRegression',extracted)

    def test_real_package_whitelist_and_baseline_preservation(self):
        root=Path('artifacts/improvement-v39/dataset')
        if not root.exists():self.skipTest('Audited artifacts not built yet')
        expected={'adapter/adapter_model.safetensors','adapter/adapter_config.json','readout.npz','training-result.json','readout-result.json','training-audit.json','component_inference.py','v39-asset-manifest.json','dataset-metadata.json'}
        self.assertEqual({str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()},expected)
        manifest=json.loads((root/'v39-asset-manifest.json').read_text())
        for item in manifest['files']:
            path=root/item['path'];self.assertEqual(path.stat().st_size,item['bytes'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),item['sha256'])
        baseline=nbformat.read('notebooks/improvement-v16/name-that-llm-two-stage-v16.ipynb',4)
        candidate=nbformat.read('notebooks/improvement-v39/inference.ipynb',4)
        old=[c.source for c in baseline.cells if c.cell_type=='code']
        new=[c.source for c in candidate.cells if c.cell_type=='code']
        self.assertEqual([s for s in new if s in old],old[:-1])
        self.assertEqual(sum("vote('ce8b_v39',softmax(scores),0.15)" in s for s in new),1)
        metadata=json.loads(Path('notebooks/improvement-v39/kernel-metadata.json').read_text())
        self.assertEqual(metadata['kernel_sources'],[])
        self.assertNotIn('qwen-lm/qwen-3/Transformers/4b-base/1',metadata['model_sources'])
        self.assertTrue(metadata['enable_gpu'] and metadata['is_private'])
        self.assertFalse(metadata['enable_internet'])

    def test_quicksave_serialization_retains_all_attachment_fields(self):
        request=QuickSaveRequest()
        fields={'dataset_data_sources':['owner/private-data'],'model_data_sources':['owner/model/Transformers/base/1'],'competition_data_sources':['name-that-llm'],'kernel_data_sources':[]}
        for name,value in fields.items():setattr(request,name,value)
        payload=request.to_field_map(request,ignore_defaults=True)
        for name,value in fields.items():self.assertEqual(payload[name],value)
        self.assertEqual(payload['kernel_execution_type'],'QuickSave')
        self.assertEqual(payload['machine_shape'],'NvidiaTeslaT4')
        a=json.dumps({'cells':[{'cell_type':'code','source':['x=1\n','print(x)']}]})
        b=json.dumps({'cells':[{'cell_type':'code','source':'x=1\nprint(x)'}]})
        self.assertEqual(cell_sources(a),cell_sources(b))

    def test_missing_audit_fails_before_creating_output(self):
        cwd=Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                with self.assertRaisesRegex(AssertionError,'Full-host audit not retrieved'):build()
                self.assertFalse(Path('artifacts/improvement-v39').exists())
            finally:os.chdir(cwd)

if __name__=='__main__':unittest.main()
