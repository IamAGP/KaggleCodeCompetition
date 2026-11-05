"""Run the frozen attributed comparison through the official Kaggle SDK."""
import argparse
import hashlib
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest
from launch_v39_nov5 import SaveAndRunAllRequest, cell_sources, normalize_metadata


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    folder=Path('notebooks/nov5-public-control');out=Path('artifacts/nov5-public-control');out.mkdir(exist_ok=True)
    meta=json.loads((folder/'kernel-metadata-ce8b.json').read_text())
    manifest=json.loads((folder/'build-manifest-ce8b.json').read_text())
    source=(folder/meta['code_file']).read_text()
    digest=hashlib.sha256(source.encode()).hexdigest()
    assert digest==manifest['candidate_sha256']=='59639ba6b606adc88da876facab09ac9f3859fc984bc9eea7fb686e3e0a5955a'
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert len(json.loads(source)['cells'])==22
    request=SaveAndRunAllRequest();request.slug=meta['id'];request.new_title=meta['title'];request.text=source
    request.language='python';request.kernel_type='notebook';request.is_private=True;request.enable_gpu=True
    request.enable_tpu=False;request.enable_internet=False;request.session_timeout_seconds=32400
    mapping={'dataset_data_sources':'dataset_sources','kernel_data_sources':'kernel_sources','competition_data_sources':'competition_sources','model_data_sources':'model_sources'}
    for field,key in mapping.items():setattr(request,field,meta[key])
    payload=request.to_field_map(request,ignore_defaults=True)
    assert payload['docker_image']==meta['docker_image'] and payload['docker_image_pinning_type']=='original'
    if not args.execute and not args.verify_only:
        print(json.dumps({k:v for k,v in payload.items() if k!='text'},indent=2));return
    receipt_path=out/'save-receipt.json'
    if not args.verify_only: assert not receipt_path.exists(),'Prior launch exists; inspect authoritative status before any new version'
    api=KaggleApi();api.authenticate()
    with api.build_kaggle_client() as client:
        if args.verify_only:
            saved=json.loads(receipt_path.read_text())
        else:
            saved=client.kernels.kernels_api_client.save_kernel(request).to_dict()
            receipt_path.write_text(json.dumps(saved,indent=2))
        assert not saved.get('error'),saved
        canonical=saved['ref'].removeprefix('/code/')
        query=ApiGetKernelRequest();query.user_name=canonical.split('/')[0];query.kernel_slug=canonical.split('/')[1]
        remote=client.kernels.kernels_api_client.get_kernel(query)
    actual=normalize_metadata(remote.metadata.to_dict())
    (out/'saved-metadata.json').write_text(json.dumps(actual,indent=2))
    (out/'saved-source.ipynb').write_text(remote.blob.source)
    assert cell_sources(source)==cell_sources(remote.blob.source)
    def normalized(ref,field):
        return '/'.join(ref.lower().split('/')[:2]) if field=='dataset_data_sources' else ref.lower()
    for field,key in mapping.items():
        assert sorted(normalized(v,field) for v in actual.get(field,[]))==sorted(normalized(v,field) for v in meta[key]),(field,actual.get(field))
    assert actual['is_private'] and actual['enable_gpu'] and not actual.get('enable_internet',False)
    result={'save':saved,'source_sha256':digest,'source_verified':True,'all_attachment_identities_verified':True,'dataset_versions_requested':meta['dataset_sources'],'version_resolution_note':'SDK readback may normalize dataset refs; exact pool bytes/hash checked at runtime. Kernel-output refs observed versions captured in source manifest.','executed':False,'submitted':False}
    (out/'launch-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
