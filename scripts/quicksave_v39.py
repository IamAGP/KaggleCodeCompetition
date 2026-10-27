"""Private V39 QuickSave through official SDK; preserve and read back all inputs."""
import argparse
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi, ApiSaveKernelRequest, ApiGetKernelRequest


class QuickSaveRequest(ApiSaveKernelRequest):
    @staticmethod
    def to_field_map(value,ignore_defaults=False):
        payload=ApiSaveKernelRequest.to_field_map(value,ignore_defaults=ignore_defaults)
        payload['kernel_execution_type']='QuickSave'
        payload['machine_shape']='NvidiaTeslaT4'
        for key in ['dataset_data_sources','kernel_data_sources','competition_data_sources','model_data_sources']:
            payload[key]=list(getattr(value,key))
        return payload


def cell_sources(source):
    notebook=json.loads(source)
    return [(c['cell_type'],''.join(c['source']) if isinstance(c['source'],list) else c['source']) for c in notebook['cells']]


def main(execute=False):
    root=Path('artifacts/improvement-v39');folder=Path('notebooks/improvement-v39')
    built=json.loads((root/'build-receipt.json').read_text());source=(folder/'inference.ipynb').read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==built['notebook_sha256']
    metadata=json.loads((folder/'kernel-metadata.json').read_text())
    body=json.loads(source)
    for cell in body['cells']:
        if cell['cell_type']=='code':assert cell['outputs']==[] and cell['execution_count'] is None
        if isinstance(cell['source'],list):cell['source']=''.join(cell['source'])
    request=QuickSaveRequest();request.slug=metadata['id'];request.new_title=metadata['title'];request.text=json.dumps(body)
    request.language='python';request.kernel_type='notebook';request.is_private=True;request.enable_gpu=True
    request.enable_tpu=False;request.enable_internet=False;request.session_timeout_seconds=32400
    mapping={'dataset_data_sources':'dataset_sources','kernel_data_sources':'kernel_sources','competition_data_sources':'competition_sources','model_data_sources':'model_sources'}
    for field,key in mapping.items():setattr(request,field,metadata[key])
    payload=request.to_field_map(request,ignore_defaults=True)
    for field,key in mapping.items():assert payload[field]==metadata[key]
    assert payload['kernel_execution_type']=='QuickSave'
    if not execute:
        print(json.dumps({k:v for k,v in payload.items() if k!='text'},indent=2));return
    readback=json.loads((root/'private-dataset-readback-receipt.json').read_text())
    assert readback['all_nine_match'] and len(readback['verified_files'])==9
    assert not (root/'sdk-quicksave-receipt.json').exists(),'Refuse duplicate save without explicit new review'
    api=KaggleApi();api.authenticate()
    with api.build_kaggle_client() as client:
        result=client.kernels.kernels_api_client.save_kernel(request)
        record=result.to_dict();(root/'sdk-quicksave-receipt.json').write_text(json.dumps(record,indent=2))
        query=ApiGetKernelRequest();query.user_name=metadata['id'].split('/')[0];query.kernel_slug=metadata['id'].split('/')[1]
        remote=client.kernels.kernels_api_client.get_kernel(query)
    actual=remote.metadata.to_dict();(root/'saved-kernel-metadata.json').write_text(json.dumps(actual,indent=2))
    (root/'saved-notebook.ipynb').write_text(remote.blob.source)
    assert cell_sources(remote.blob.source)==cell_sources(source),'Saved notebook source changed'
    for field,key in mapping.items():
        values=actual.get(field) or []
        assert sorted(v.lower() for v in values)==sorted(v.lower() for v in metadata[key]),(field,values)
    assert actual['is_private'] is True and actual['enable_gpu'] is True
    assert actual.get('enable_internet',False) is False and actual.get('enable_tpu',False) is False
    receipt={'private':True,'all_attachments_verified':True,'source_cells_verified':True,
             'notebook_sha256':built['notebook_sha256'],'save_response':record,
             'arrays':{field:actual.get(field) or [] for field in mapping},
             'execution_proven':False,'submission_performed':False,'mode':'QuickSave'}
    (root/'private-save-verification.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');main(parser.parse_args().execute)
