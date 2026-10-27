"""Save reviewed V37 through official SDK with explicit QuickSave and inputs."""
import argparse
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi, ApiSaveKernelRequest


class QuickSaveRequest(ApiSaveKernelRequest):
    @staticmethod
    def to_field_map(value, ignore_defaults=False):
        data=ApiSaveKernelRequest.to_field_map(value,ignore_defaults=ignore_defaults)
        # These supported API fields are absent from the installed older SDK model.
        data['kernel_execution_type']='QuickSave'
        data['machine_shape']='NvidiaTeslaT4'
        return data


def main(execute=False):
    folder=Path('notebooks/improvement-v37')
    source=(folder/'inference.ipynb').read_text()
    assert hashlib.sha256(source.encode()).hexdigest()=='d552dee506df3dc7f82ea9251d47a959c3233397ba6b82c03952a5800105e625'
    m=json.loads((folder/'kernel-metadata.json').read_text())
    r=QuickSaveRequest();r.id=141150795;r.slug=m['id'];r.new_title=m['title'];r.text=source
    r.language='python';r.kernel_type='notebook';r.is_private=True;r.enable_gpu=True
    r.enable_tpu=False;r.enable_internet=False;r.session_timeout_seconds=32400
    r.dataset_data_sources=m['dataset_sources'];r.kernel_data_sources=m['kernel_sources']
    r.competition_data_sources=m['competition_sources'];r.model_data_sources=m['model_sources']
    payload=r.to_field_map(r,ignore_defaults=True)
    assert payload['kernel_execution_type']=='QuickSave'
    for key in ['dataset_data_sources','kernel_data_sources','competition_data_sources','model_data_sources']:
        assert payload[key],key
    if not execute:
        print(json.dumps({k:v for k,v in payload.items() if k!='text'},indent=2));return
    api=KaggleApi();api.authenticate()
    with api.build_kaggle_client() as client:
        result=client.kernels.kernels_api_client.save_kernel(r)
    record=result.to_dict()
    Path('artifacts/improvement-v37/sdk-quicksave-receipt.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    main(parser.parse_args().execute)
