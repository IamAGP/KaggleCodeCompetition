"""Freeze V40 inputs and settings without changing any earlier experiment."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path('artifacts/improvement-v40')
    root.mkdir(exist_ok=False)
    inputs = root/'inputs'
    inputs.mkdir()
    shutil.copy2('data/name-that-llm/train.csv', inputs/'train.csv')
    shutil.copy2('artifacts/baseline-v1/folds.csv', inputs/'folds.csv')
    origin = json.loads(Path('artifacts/improvement-v34/frozen-plan.json').read_text())
    prior = origin['arms']['4b_small_supcon']
    checkpoint = Path('artifacts/improvement-v18/kaggle-output/ce_supcon/external_only')
    for name,digest in prior['checkpoint_sha256'].items():
        assert sha(checkpoint/name) == digest
    shutil.copytree(checkpoint,inputs/'external')
    assert sha(inputs/'train.csv') == origin['host_sha256']
    plan = {
        'experiment':'improvement-v40', 'status':'prepared_not_launched',
        'hypothesis':'A class-restricted nonlinear representation may improve attribution, with and without supervised contrastive loss.',
        'base_model':'Qwen/Qwen3-4B-Base', 'initialization':'V18 ce_supcon external-only adapter and LayerNorm; all new heads freshly initialized identically where shared',
        'base_architecture':{'layers':36,'hidden_size':2560},
        'quantization':'NF4 double quantization, BF16 computation',
        'lora':{'rank':16,'alpha':32,'target_modules':['q_proj','v_proj'],'layers':'all 36'},
        'pooling':'attention-masked mean of all response tokens plus EOS; no first-token removal',
        'head':'LayerNorm(2560) -> Linear(2560,1800) -> GELU; dense Linear(1800,18) or 18 separate Linear(100,1) heads',
        'projection':'Linear(1800,256) -> GELU -> Linear(256,128); L2 normalization in SupCon',
        'arms':['dense_ce','blocks_ce','dense_supcon','blocks_supcon'],
        'folds':[1,2], 'labels':origin['labels'], 'seed':4001,
        'epochs':2, 'batch_size':4, 'accumulation':8, 'max_length':256,
        'adapter_lr':3e-5, 'head_lr':1e-4, 'weight_decay':0.01,
        'temperature':0.1,'contrastive_weight':0.1,
        'sampling':'Identical crossed 2 authors x 2 prompts in all arms, deterministic per epoch and fold. Gradient accumulation does not enlarge contrastive batch.',
        'schedule':'fold 1 then fold 2; all four arms each; no interim winner selection',
        'evaluation':'Final epoch direct head, raw balanced accuracy; no posthoc readout or balance tuning',
        'gate':{'blocks_supcon_over_dense_supcon_pooled_min':0.01,'both_fold_gains_positive':True,'action':'broader confirmation and incremental ensemble evaluation, not automatic promotion'},
        'prior_evidence':{'v35_pooled_independent':0.43333333333333335,'v35_pooled_crossed':0.425925925925926,'interpretation':'Sampling-only comparison failed. V40 tests new structured nonlinear heads and their interaction with loss; it does not repeat V35.'},
        'per_fit_seconds':1500,'round_training_seconds':12600,
        'budget':{'existing_user_cap_usd':10,'maximum_new_spend_usd':6,'launch_requires':'live opening balance, prior spend reconciliation, exact rate, provider-capable watchdog active before creation; no billed resource created by this preparation'},
        'inputs':{str(p.relative_to(inputs)):sha(p) for p in inputs.rglob('*') if p.is_file()},
        'sources':{
            'qwen':'https://arxiv.org/html/2505.09388v1',
            'qwen_code':'https://github.com/QwenLM/Qwen3',
            'supcon':'https://arxiv.org/html/2004.11362v5',
            'supcon_code':'https://github.com/HobbitLong/SupContrast/blob/master/losses.py',
            'quantization':'https://huggingface.co/docs/peft/en/developer_guides/quantization'},
        'limitations':['Existing host labels and splits have been inspected previously; screening evidence only.',
            'Dense and block readouts differ by 30,600 parameters; common 4.6M-parameter nonlinear map is identical.',
            'Blocks are separate, not nested Matryoshka prefixes.',
            'External-only initialization already used SupCon, shared across all four arms.']}
    files=['model_blocks_v40.py','train_blocks_v40.py','sampling_v35.py','run_blocks_v40.py','evaluate_blocks_v40.py']
    code=root/'code';code.mkdir()
    for name in files: shutil.copy2(Path('scripts')/name,code/name)
    plan['code_sha256']={name:sha(code/name) for name in files}
    target=Path('configs/name-that-llm/v40.json')
    assert not target.exists()
    target.write_text(json.dumps(plan,indent=2)+'\n')
    (root/'plan.json').write_text(target.read_text())
    print(json.dumps({'prepared':str(root),'plan_sha256':sha(target),'input_bytes':sum(p.stat().st_size for p in inputs.rglob('*') if p.is_file()),'launched':False}))

if __name__=='__main__':main()
