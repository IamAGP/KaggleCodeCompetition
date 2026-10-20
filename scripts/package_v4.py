"""Package v4 source, validation evidence and any scored notebook for private S3."""

import json
from pathlib import Path

from package_v3 import bundle, digest


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "artifacts/improvement-v4"
    source = [root / n for n in (".gitignore", "AGENTS.md", "README.md", "pyproject.toml")]
    for folder in ("src", "scripts", "tests", "notebooks", "configs"):
        source.extend(p for p in (root / folder).rglob("*")
                      if p.is_file() and p.suffix in {".py", ".json", ".ipynb"}
                      and "__pycache__" not in p.parts and not p.name.endswith("executed.ipynb"))
    bundle(run / "source.zip", source, root)
    excluded = {"archive-manifest.json", "run-manifest.json", "run-bundle.zip"}
    files = sorted(p for p in run.rglob("*")
                   if p.is_file() and p.name not in excluded and "__pycache__" not in p.parts)
    inventory = {"files": [{"path": str(p.relative_to(run)), "bytes": p.stat().st_size,
                            "sha256": digest(p)} for p in files]}
    (run / "run-manifest.json").write_text(json.dumps(inventory, indent=2) + "\n")
    bundle(run / "run-bundle.zip", files + [run / "run-manifest.json"], run)
    direct = {n: run / n for n in ("run-bundle.zip", "source.zip", "run-manifest.json")}
    direct["README.md"] = root / "README.md"
    optional = {
        "kaggle_submission.json": run / "kaggle_submission.json",
        "checkpoint.bundle": run / "checkpoint.bundle",
        "checkpoint.json": run / "checkpoint.json",
        "name-that-llm-nested-stack-v4.ipynb":
            root / "notebooks/improvement-v4/name-that-llm-nested-stack-v4.ipynb",
    }
    direct.update({n: p for n, p in optional.items() if p.exists()})
    prefix = "name-that-llm/experiments/improvement-v4/"
    manifest = {
        "bucket": "kaggle-featured-code-comps", "region": "ap-south-1",
        "objects": [{"local_path": str(p), "key": prefix + n, "size_bytes": p.stat().st_size,
                     "sha256": digest(p)} for n, p in direct.items()],
    }
    (run / "archive-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(source)} source files and {len(files)} experiment files; "
          f"{len(direct)} objects ready for S3.")


if __name__ == "__main__":
    main()
