"""Reuse audited v21 full-host training with the qualified v22 external checkpoint."""
import hashlib
import json
from pathlib import Path


def build():
    audit=json.loads(Path('artifacts/improvement-v22/evaluation-audit.json').read_text())
    assert audit['verified'] and audit['selected_for_full_host_training']=='supcon_final'
    config=json.loads(Path('configs/name-that-llm/v23.json').read_text())
    source=Path('scripts/build_training_v21.py').read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==config['packaging_template_sha256']
    replacements={
        'configs/name-that-llm/v21.json':'configs/name-that-llm/v23.json',
        'artifacts/improvement-v18/kaggle-output':'artifacts/improvement-v22/kaggle-output',
        "=='improvement-v18'":"=='improvement-v22'",
        'notebooks/training-v21':'notebooks/training-v23',
        'artifacts/improvement-v21':'artifacts/improvement-v23',
        'adithyagiri/name-that-llm-contrastive-v18':'adithyagiri/name-that-llm-scaled-training-v22',
        'adithyagiri/name-that-llm-full-contrastive-v21':'adithyagiri/name-that-llm-full-scaled-v23',
        'Name That LLM Full Contrastive V21':'Name That LLM Full Scaled V23',
        '# V21: full-host contrastive component for an additive submission':'# V23: larger-data contrastive replacement component',
        'our v18 external-only checkpoint':'our qualified v22 external-only checkpoint',
        "'v21_training'":"'v23_training'",
    }
    for old,new in replacements.items():
        assert old in source,old
        source=source.replace(old,new)
    namespace={'__name__':'v23_packaging'}
    exec(compile(source,'v21-packaging-reused-for-v23','exec'),namespace)
    namespace['build']()


if __name__=='__main__':build()
