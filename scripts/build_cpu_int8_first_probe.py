"""Measure INT8 throughput first; no numerical-parity claim."""
import json,hashlib
from pathlib import Path
import nbformat
root=Path('artifacts/cpu-int8-first-probe');root.mkdir(exist_ok=True)
old=Path('artifacts/cpu-stream-int8-probe/worker.py').read_text()
s=old.replace("PHASE='fp16_reference'","PHASE='int8_dynamic'",1)
a=s.index('    for s in range(0,len(order),bs):embed(order[s:s+bs])')
b=s.index("    PHASE='int8_dynamic'",a)
s=s[:a]+s[b:]
a=s.index("with np.load('fp16_reference.npz',allow_pickle=False) as ref:")
s=s[:a]+'''seconds=sum(x['seconds'] for x in FORWARD_TIMES)
per_row=seconds/len(test)
criterion='promising' if per_row<=3 else ('no_go_single_member_exceeds_budget' if per_row>8 else 'inconclusive')
assessment={'seconds_per_row':per_row,'criterion':criterion,'fixed_promising_seconds_per_row':3,'fixed_no_go_seconds_per_row':8,'rows':len(test),'scope':'Three visible rows,203paddedtokens expected; one14Bmember only. No quantization parity or hidden runtime guarantee.'}
Path('throughput-assessment.json').write_text(json.dumps(assessment,indent=2));print(json.dumps({'event':'throughput_assessment',**assessment}),flush=True)
'''
s=s.replace("'limit':600 if PHASE=='fp16_reference' else 300","'limit':300")
compile(s,'int8first','exec');(root/'worker.py').write_text(s)
nb=nbformat.read('artifacts/cpu-stream-int8-probe/probe.ipynb',4)
runner=nb.cells[1].source.replace(repr(old),repr(s)).replace('elapsed>1800','elapsed>1200').replace('limit=600','limit=300')
assert runner!=nb.cells[1].source
nb.cells[1].source=runner;nb.cells[0].source='# CPU INT8 throughput-first probe\nValidated streaming FP16 loading and originalLoRA merge, immediate INT8 conversion. Mandatorytinyfixture.27GiB RSS/cgroup,300sforward,1200stotal. Fixed thresholds≤3s/row promising,>8s/row no-go. Three visible rows; no parityclaim or submission.'
compile(runner,'runner','exec');nbformat.validate(nb);nbformat.write(nb,root/'probe.ipynb')
m=json.loads(Path('artifacts/cpu-stream-int8-probe/kernel-metadata.json').read_text());m.update(id='adithyagiri/name-that-llm-cpu-int8-throughput-first',title='Name That LLM CPU INT8 Throughput First')
(root/'kernel-metadata.json').write_text(json.dumps(m,indent=2))
manifest={'worker_sha256':hashlib.sha256(s.encode()).hexdigest(),'notebook_sha256':hashlib.sha256((root/'probe.ipynb').read_bytes()).hexdigest(),'limits':{'whole_seconds':1200,'forward_seconds':300,'rss_and_cgroup_gib':27},'go_no_go':{'promising_seconds_per_row_lte':3,'no_go_seconds_per_row_gt':8},'parity_claim':False,'previous_attempt_preserved':True,'fixture_retained':True,'scope':'Same3publicrows, expected203paddedtokens; no othermembers, no hidden runtime guarantee'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
