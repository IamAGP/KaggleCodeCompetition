"""Package source and enumerate baseline artifacts for private S3 archival."""

import hashlib
import json
from pathlib import Path
import zipfile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    run = root / "artifacts/baseline-v1"
    source_files = [root / name for name in (
        ".gitignore", "AGENTS.md", "README.md", "pyproject.toml",
    )]
    for folder in ("src", "scripts", "tests", "notebooks"):
        source_files.extend(
            path for path in (root / folder).rglob("*")
            if path.is_file() and path.suffix in {".py", ".ipynb", ".json"}
            and "__pycache__" not in path.parts
            and not path.name.endswith("executed.ipynb")
        )
    archive = run / "source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(source_files):
            bundle.write(path, str(path.relative_to(root)))
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Source archive integrity check failed.")
        for path in source_files:
            if bundle.read(str(path.relative_to(root))) != path.read_bytes():
                raise RuntimeError(f"Archived source differs: {path}")
    files = sorted(
        path for path in run.rglob("*")
        if path.is_file() and path.name != "archive-manifest.json"
    )
    files += [
        root / "notebooks/name-that-llm-baseline.ipynb",
        root / "notebooks/kernel-metadata.json",
    ]
    objects = []
    for path in files:
        relative = path.relative_to(run) if path.is_relative_to(run) else Path(path.name)
        data = path.read_bytes()
        objects.append({
            "local_path": str(path),
            "key": "name-that-llm/baselines/v1/" + str(relative),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    manifest = {
        "bucket": "kaggle-featured-code-comps", "region": "ap-south-1",
        "objects": objects,
    }
    (run / "archive-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Packaged {len(source_files)} source files; {len(objects)} artifacts for S3.")


if __name__ == "__main__":
    main()
