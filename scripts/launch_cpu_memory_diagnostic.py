"""Private CPU-only cgroup memory diagnostic; no model/data, max120s."""
import hashlib,json
from pathlib import Path
import nbformat
from kaggle.api.kaggle_api_extended import KaggleApi,ApiSaveKernelRequest
class Request(ApiSaveKernelRequest):
    @staticmethod
    def to_field_map(value,ignore_defaults=False):
        data=ApiSaveKernelRequest.to_field_map(value,ignore_defaults=ignore_defaults)
        data['kernel_execution_type']='SaveAndRunAll';return data
root=Path('artifacts/cpu-memory-diagnostic');root.mkdir(exist_ok=True)
worker='''import json,os,resource,signal
from pathlib import Path
signal.alarm(60)
base=Path('/sys/fs/cgroup')
paths=[base,base/'memory']
for line in Path('/proc/self/cgroup').read_text().splitlines():
    _,controllers,path=line.split(':',2)
    candidate=(base/'memory'/path.lstrip('/')) if 'memory' in controllers.split(',') else base/path.lstrip('/')
    if '..' not in candidate.parts:
        while candidate.is_relative_to(base):
            if candidate not in paths:paths.append(candidate)
            if candidate==base:break
            candidate=candidate.parent
records=[]
for p in paths:
    values={}
    for name in ['memory.max','memory.high','memory.current','memory.peak','memory.events','memory.limit_in_bytes','memory.usage_in_bytes','memory.max_usage_in_bytes','memory.failcnt']:
        file=p/name
        if file.is_file():
            try:values[name]=file.read_text().strip()
            except OSError as e:values[name]={'error':type(e).__name__}
    if values:records.append({'scope':'cgroup_root' if p==base else 'memory_hierarchy','values':values})
meminfo={}
for line in Path('/proc/meminfo').read_text().splitlines():
    key,value=line.split(':',1)
    if key in ['MemTotal','MemAvailable','MemFree','SwapTotal','SwapFree']:meminfo[key]=value.strip()
result={'cgroups':records,'meminfo':meminfo,'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'logical_cpus':os.cpu_count(),'scope':'Fresh CPU notebook capacity, not previous session peak or guarantee of future placement'}
Path('memory.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
signal.alarm(0)
'''
compile(worker,'memory','exec');(root/'worker.py').write_text(worker)
nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(worker)],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
nbformat.write(nb,root/'probe.ipynb');source=(root/'probe.ipynb').read_text()
r=Request();r.slug='adithyagiri/name-that-llm-cpu-memory-diagnostic';r.new_title='Name That LLM CPU Memory Diagnostic';r.text=source;r.language='python';r.kernel_type='notebook';r.is_private=True;r.enable_gpu=False;r.enable_tpu=False;r.enable_internet=False;r.session_timeout_seconds=120
r.dataset_data_sources=[];r.model_data_sources=[];r.kernel_data_sources=[];r.competition_data_sources=[]
(root/'preflight.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'private':True,'gpu':False,'internet':False,'session_timeout_seconds':120,'worker_timeout_seconds':60,'attachments':[]},indent=2))
api=KaggleApi();api.authenticate()
with api.build_kaggle_client() as client:result=client.kernels.kernels_api_client.save_kernel(r)
receipt=result.to_dict();(root/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
