"""Replace the v21 component with the qualified larger-data v23 component."""
from pathlib import Path


def build():
    source=Path('scripts/build_notebook_v21.py').read_text()
    replacements={
        'artifacts/improvement-v21':'artifacts/improvement-v23',
        'notebooks/improvement-v21':'notebooks/improvement-v23',
        "=='improvement-v21'":"=='improvement-v23'",
        'adithyagiri/name-that-llm-full-contrastive-v21':'adithyagiri/name-that-llm-full-scaled-v23',
        'adithyagiri/name-that-llm-contrastive-blend-v21':'adithyagiri/name-that-llm-scaled-blend-v23',
        'Name That LLM Contrastive Blend V21':'Name That LLM Scaled Blend V23',
        '# V21: prospective contrastive addition to the 0.620 ensemble':'# V23: larger-data replacement of the v21 contrastive component',
        'contrastive_v21':'contrastive_v23',
        'v21_scores.npz':'v23_scores.npz',
        'v21_audit.json':'v23_audit.json',
        "'experiment':'v21'":"'experiment':'v23'",
        'our v18/v19/v21 experiments':'our v18/v19/v22/v23 experiments',
    }
    for old,new in replacements.items():
        assert old in source,old
        source=source.replace(old,new)
    namespace={'__name__':'v23_inference_packaging'}
    exec(compile(source,'v21-inference-reused-for-v23','exec'),namespace)
    namespace['build']()


if __name__=='__main__':build()
