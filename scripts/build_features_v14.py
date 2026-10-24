"""Create private, label-free Qwen3-8B feature extraction on Kaggle."""
import json
import hashlib
from pathlib import Path
import nbformat

CODE = r'''
import os, glob, json, time, hashlib, gc
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
T0 = time.time()
LAYERS = [9, 24]
def log(message):
    print(f'[{time.time()-T0:.1f}s] {message}', flush=True)
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
train, test = pd.read_csv(DATA + '/train.csv'), pd.read_csv(DATA + '/test.csv')
paths = glob.glob('/kaggle/input/**/config.json', recursive=True)
paths = [p for p in paths if '/8b-base/' in p.lower()]
assert len(paths) == 1, paths
MODEL_DIR = os.path.dirname(paths[0])
assert torch.cuda.is_available(), 'GPU required'
log({'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
tok.padding_side = 'right'
model = AutoModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float16,
    device_map='balanced', max_memory={i:'12GiB' for i in range(torch.cuda.device_count())},
    local_files_only=True).eval()
assert model.config.num_hidden_layers == 36 and model.config.hidden_size == 4096
log({'device_map': str(model.hf_device_map)})
@torch.inference_mode()
def encode(texts, name):
    texts = list(texts)
    order = np.argsort([len(t) for t in texts], kind='stable')
    features = np.zeros((len(texts), 2, 2, 4096), dtype=np.float32)
    for start in range(0, len(texts), 8):
        ids = order[start:start+8]
        enc = tok([texts[i] for i in ids], return_tensors='pt', padding=True,
                  truncation=True, max_length=512)
        enc = {k:v.to(model.device) for k,v in enc.items()}
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        attention = enc['attention_mask']
        for j, layer in enumerate(LAYERS):
            h = hs[layer].float()
            full = attention.to(h.device)
            prefix = full * (full.cumsum(1) <= 128)
            for pool, mask in enumerate((full, prefix)):
                m = mask.unsqueeze(-1).float()
                features[ids,pool,j] = ((h*m).sum(1)/m.sum(1).clamp_min(1)).cpu().numpy()
        del hs
        if start % 200 == 0:
            log({'split':name,'completed':min(start+8,len(texts)),'total':len(texts)})
    assert np.isfinite(features).all()
    np.save(f'v14_{name}_features.npy', features)
    return features
train_features = encode(train.text.fillna('').astype(str), 'train')
test_features = encode(test.text.fillna('').astype(str), 'test')
report = {'model':'qwen-lm/qwen-3/Transformers/8b-base/1', 'layers':LAYERS,
          'pooling':['mean512','prefix128'], 'train_ids':train.id.tolist(),
          'test_ids':test.id.tolist(), 'train_csv_sha256':hashlib.sha256(open(DATA+'/train.csv','rb').read()).hexdigest(),
          'model_config':model.config.to_dict(), 'seconds':time.time()-T0,
          'shape':list(train_features.shape),'torch':torch.__version__}
with open('v14_features_report.json','w') as f: json.dump(report,f,indent=2)
log({'finished':True,'seconds':report['seconds']})
'''

def build():
    folder=Path('notebooks/features-v14');folder.mkdir(exist_ok=True)
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(
        '# Qwen3-8B frozen features V14\n\nLabel-free feature extraction for separately frozen prompt-held-out evaluation. '
        'Mean of layers 9/24 over up to 512 tokens and first 128 tokens. No fine-tuning or test-group assumptions.'),
        nbformat.v4.new_code_cell(CODE.strip())])
    nb.metadata['kernelspec'] = {'display_name':'Python 3','language':'python','name':'python3'}
    compile(CODE,'<features-v14>','exec');nbformat.validate(nb)
    nbformat.write(nb,folder/'features.ipynb')
    meta={'id':'adithyagiri/name-that-llm-qwen8b-features-v14',
          'title':'Name That LLM Qwen8B Features V14','code_file':'features.ipynb',
          'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,
          'machine_shape':'NvidiaTeslaT4','enable_internet':False,'enable_tpu':False,
          'competition_sources':['name-that-llm'],'dataset_sources':[], 'kernel_sources':[],
          'model_sources':['qwen-lm/qwen-3/Transformers/8b-base/1']}
    (folder/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(folder)

if __name__=='__main__': build()
