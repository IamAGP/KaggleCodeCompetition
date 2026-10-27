"""Run the reviewed bounded private CPU probe through the official Kaggle SDK."""
import hashlib
import json
from pathlib import Path
import nbformat
from kaggle.api.kaggle_api_extended import KaggleApi,ApiSaveKernelRequest


class CpuProbeRequest(ApiSaveKernelRequest):
    @staticmethod
    def to_field_map(value,ignore_defaults=False):
        data=ApiSaveKernelRequest.to_field_map(value,ignore_defaults=ignore_defaults)
        data['kernel_execution_type']='SaveAndRunAll'
        return data


def main():
    root=Path('artifacts/cpu-int8-first-probe')
    notebook=root/'probe.ipynb';source=notebook.read_text()
    nb=nbformat.read(notebook,4)
    assert 'elapsed>1200 or forward_elapsed>limit or rss>27*2**30 or cgroup_current>27*2**30' in nb.cells[1].source
    assert nb.metadata.kernelspec.name=='python3'
    worker=(root/'worker.py').read_text()
    assert 'assert len(test)==3' in worker
    assert 'int8_member_scores.npz' in worker
    assert 'dtype = torch.float16' in worker
    assert 'quantized.dynamic.Linear.from_float' in worker
    assert "PHASE='int8_dynamic'" in worker
    assert 'fp16_reference.npz' not in worker
    assert 'submission.parquet' not in worker
    assert worker.index('validate_sharded_fixture(merge_lora)\ngc.collect()') < worker.index("ckpt_name='lora_qwen3_14b'")
    compile(worker,'cpu-probe-worker','exec')
    metadata=json.loads((root/'kernel-metadata.json').read_text())
    assert metadata['is_private'] and not metadata['enable_gpu'] and not metadata['enable_internet']
    request=CpuProbeRequest();request.slug=metadata['id'];request.new_title=metadata['title']
    request.text=source;request.language='python';request.kernel_type='notebook'
    request.is_private=True;request.enable_gpu=False;request.enable_tpu=False;request.enable_internet=False
    request.session_timeout_seconds=1200
    request.dataset_data_sources=metadata['dataset_sources'];request.model_data_sources=metadata['model_sources']
    request.kernel_data_sources=metadata['kernel_sources'];request.competition_data_sources=metadata['competition_sources']
    preflight={'notebook_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'worker_sha256':hashlib.sha256(worker.encode()).hexdigest(),'metadata':metadata,
        'session_timeout_seconds':1200,'execution_type':'SaveAndRunAll','competition_submission':False}
    (root/'launch-preflight.json').write_text(json.dumps(preflight,indent=2))
    api=KaggleApi();api.authenticate()
    with api.build_kaggle_client() as client:result=client.kernels.kernels_api_client.save_kernel(request)
    receipt=result.to_dict();(root/'launch-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':main()
