"""Apply the unchanged v21 full-host artifact checks to v23 outputs."""
from pathlib import Path


if __name__=='__main__':
    source=Path('scripts/audit_v21_training.py').read_text()
    assert "artifacts/improvement-v21" in source
    source=source.replace('artifacts/improvement-v21','artifacts/improvement-v23')
    namespace={'__name__':'v23_audit'}
    exec(compile(source,'v21-audit-reused-for-v23','exec'),namespace)
    namespace['audit']()
