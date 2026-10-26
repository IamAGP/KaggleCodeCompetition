"""Apply fixed V25 gates with V22 data/score checks and 8B adapter dimensions."""
from pathlib import Path


def main():
    source = Path('scripts/evaluate_v22.py').read_text()
    changes = {
        "root=Path('artifacts/improvement-v22')": "root=Path('artifacts/improvement-v25')",
        "'readout_worker_sha256','packaging_template_sha256'":
            "'readout_worker_sha256','packaging_template_sha256','feasibility_evidence'",
        "pd.read_parquet(root/'preflight'/filename)":
            "pd.read_parquet(Path('artifacts/improvement-v22/preflight')/filename)",
        'assert len(adapter)==144': 'assert len(adapter)==504',
        'x.shape==(2699,3,2560)': 'x.shape==(2699,3,4096)',
        "Path('artifacts/improvement-v19/kaggle-output')":
            "Path('artifacts/improvement-v22/kaggle-output')",
        "'small_pool_accuracy'": "'v22_accuracy'",
        "'small_pool_fixed_sparse_blend'": "'v22_fixed_sparse_blend'",
    }
    for old,new in changes.items():
        assert old in source,old
        source = source.replace(old,new)
    # Seven linear modules (q/k/v/o and gate/up/down) x 36 layers x A/B.
    needle = 'adapter=load_file(str(checkpoint))'
    source = source.replace(needle, needle+'\n        adapter_config=json.loads((checkpoint.parent/"adapter_config.json").read_text())'
        +'\n        assert adapter_config["r"]==16 and adapter_config["lora_alpha"]==32'
        +'\n        assert set(adapter_config["target_modules"])=={"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}')
    namespace = {'__name__':'v25_evaluation'}
    exec(compile(source,'v25-fixed-evaluation','exec'),namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
