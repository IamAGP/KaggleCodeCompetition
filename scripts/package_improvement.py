"""Create checksum-verified source and full-run bundles for S3."""

import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bundle_files(destination, files, base):
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(base)))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip():
            raise RuntimeError("Archive CRC check failed.")
        for path in files:
            if hashlib.sha256(archive.read(str(path.relative_to(base)))).hexdigest() != digest(path):
                raise RuntimeError(f"Archive member differs: {path}")


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "artifacts/improvement-v2"
    source = [root / name for name in (".gitignore", "AGENTS.md", "README.md", "pyproject.toml")]
    for folder in ("src", "scripts", "tests", "notebooks"):
        source.extend(path for path in (root / folder).rglob("*")
                      if path.is_file() and path.suffix in {".py", ".json", ".ipynb"}
                      and "__pycache__" not in path.parts
                      and not path.name.endswith("executed.ipynb"))
    bundle_files(run / "source.zip", source, root)
    excluded = {"archive-manifest.json", "run-manifest.json", "run-bundle.zip"}
    artifacts = sorted(path for path in run.rglob("*") if path.is_file() and path.name not in excluded)
    inventory = {
        "files": [{"path": str(p.relative_to(run)), "bytes": p.stat().st_size, "sha256": digest(p)}
                  for p in artifacts],
        "note": "Full experiment evidence, including unsuccessful candidates; excludes upload credentials.",
    }
    (run / "run-manifest.json").write_text(json.dumps(inventory, indent=2) + "\n")
    bundle_files(run / "run-bundle.zip", artifacts + [run / "run-manifest.json"], run)
    prefix = "name-that-llm/experiments/improvement-v2/"
    direct = {
        "run-bundle.zip": run / "run-bundle.zip",
        "source.zip": run / "source.zip",
        "run-manifest.json": run / "run-manifest.json",
        "selected.json": run / "selection/selected.json",
        "robustness-metrics.json": run / "robustness/metrics.json",
        "kaggle_submission.json": run / "kaggle_submission.json",
        "name-that-llm-style-ensemble-v2.ipynb": root / "notebooks/improvement-v2/name-that-llm-style-ensemble-v2.ipynb",
        "README.md": root / "README.md",
    }
    manifest = {
        "bucket": "kaggle-featured-code-comps", "region": "ap-south-1",
        "objects": [{"local_path": str(p), "key": prefix + name,
                     "size_bytes": p.stat().st_size, "sha256": digest(p)}
                    for name, p in direct.items()],
    }
    (run / "archive-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(source)} source files and {len(artifacts)} experiment files; "
          f"{len(direct)} objects ready for S3.")


if __name__ == "__main__":
    main()
