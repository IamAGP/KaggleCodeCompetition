"""Build verified private archives for v5 and its separately recorded follow-ups.

Each pinned encoder appears once in its experiment's model-bundle.zip. Run
bundles contain experiment evidence; v6 refers to v5's immutable inputs.
This script performs local packaging only. Uploads use the official AWS MCP.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile


CHUNK_SIZE = 1024 * 1024
BUCKET = "kaggle-featured-code-comps"
REGION = "ap-south-1"
BASE_PREFIX = "name-that-llm/experiments/"
GENERATED = {
    "archive-manifest.json", "run-manifest.json", "source-manifest.json",
    "model-bundle-manifest.json", "shared-inputs-manifest.json",
    "archive-verification.json", "archive-manifest-receipt.json",
    "source.zip", "run-bundle.zip", "model-bundle.zip", "checkpoint.bundle",
}
EXCLUDED_PARTS = {"model", "__pycache__", ".cache", ".git", ".aws", ".codex", ".venv", "tmp", "temp"}
SECRET_PATTERN = re.compile(
    rb"(?:\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    rb"[?&]X-Amz-(?:Signature|Credential|Security-Token)=)", re.IGNORECASE,
)


def stream_digest(stream) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(CHUNK_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return stream_digest(stream)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class ArchiveFile:
    path: Path
    name: str
    size_bytes: int
    sha256: str

    @classmethod
    def capture(cls, path: Path, name: str) -> ArchiveFile:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Archive member must be a regular local file: {path}")
        if path.name.lower() in {"credentials", "kaggle.json", "token", "auth.json"} or path.name.startswith(".env"):
            raise ValueError(f"Authentication file refused: {path}")
        if path.suffix.lower() not in {".npz", ".npy", ".parquet", ".safetensors", ".zip", ".bundle"}:
            with path.open("rb") as stream:
                previous = b""
                while chunk := stream.read(CHUNK_SIZE):
                    sample = previous + chunk
                    if SECRET_PATTERN.search(sample):
                        raise ValueError(f"Potential secret or presigned URL refused: {path}")
                    previous = sample[-512:]
        return cls(path, name, path.stat().st_size, digest(path))

    def record(self) -> dict:
        return {"path": self.name, "size_bytes": self.size_bytes, "sha256": self.sha256}


def bundle(destination: Path, members: list[ArchiveFile], *, stored: bool = False) -> None:
    """Stream file contents and verify every member against its captured digest."""
    if len({member.name for member in members}) != len(members):
        raise ValueError("Duplicate ZIP member")
    temporary = destination.with_suffix(destination.suffix + ".partial")
    compression = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(temporary, "w", compression=compression, compresslevel=None if stored else 6) as archive:
        for member in sorted(members, key=lambda item: item.name):
            info = zipfile.ZipInfo(member.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = compression
            info.external_attr = 0o100644 << 16
            with member.path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, length=CHUNK_SIZE)
    with zipfile.ZipFile(temporary) as archive:
        for member in members:
            info = archive.getinfo(member.name)
            with archive.open(info) as stream:
                saved_digest = stream_digest(stream)
            if info.file_size != member.size_bytes or saved_digest != member.sha256:
                raise RuntimeError(f"ZIP integrity or concurrent-change failure: {member.name}")
    temporary.replace(destination)


class ExperimentPackager:
    def __init__(self, root: Path, experiment: str):
        self.root = root
        self.run = root / "artifacts" / experiment
        self.experiment = experiment
        self.v5 = root / "artifacts/improvement-v5"
        if not self.run.is_dir():
            raise ValueError(f"Experiment directory does not exist: {self.run}")

    def model(self) -> None:
        manifest_path = self.run / "model/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if self.experiment == "improvement-v5" and len(manifest["files"]) != 10:
            raise ValueError("The pinned v5 model must contain exactly ten upstream files")
        members = []
        for expected in manifest["files"]:
            relative = PurePosixPath(expected["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unexpected model manifest path")
            path = self.run / "model" / relative
            member = ArchiveFile.capture(path, "model/" + expected["path"])
            if member.sha256 != expected["sha256"] or member.size_bytes != expected["size_bytes"]:
                raise ValueError(f"Pinned model differs from downloaded manifest: {path}")
            members.append(member)
        evidence = [self.run / "model/manifest.json"]
        evidence.extend(path for path in (self.run / "encoder").glob("*")
                        if path.is_file() and path.name.casefold().startswith("license"))
        if (self.run / "encoder/source-audit.json").is_file():
            evidence.append(self.run / "encoder/source-audit.json")
        if self.experiment == "improvement-v5" and not (self.run / "encoder/LICENSE.LLNL").is_file():
            raise ValueError("The reviewed v5 LLNL license is required")
        for path in sorted(evidence):
            members.append(ArchiveFile.capture(path, str(path.relative_to(self.run))))
        write_json(self.run / "model-bundle-manifest.json", {
            "model_id": manifest.get("model_id", manifest.get("repo_id")), "revision": manifest["revision"],
            "files": [member.record() for member in members],
        })
        bundle(self.run / "model-bundle.zip", members, stored=True)

    def source(self) -> list[ArchiveFile]:
        paths = [self.root / name for name in (".gitignore", "AGENTS.md", "README.md", "pyproject.toml")]
        for folder in ("src", "scripts", "tests", "notebooks", "configs"):
            paths.extend(path for path in (self.root / folder).rglob("*")
                         if path.is_file() and path.suffix in {".py", ".json", ".ipynb", ".toml"}
                         and not set(path.relative_to(self.root).parts) & EXCLUDED_PARTS
                         and not path.name.endswith("executed.ipynb"))
        members = [ArchiveFile.capture(path, str(path.relative_to(self.root))) for path in sorted(set(paths))]
        manifest = self.run / "source-manifest.json"
        write_json(manifest, {"files": [member.record() for member in members]})
        bundle(self.run / "source.zip", members + [ArchiveFile.capture(manifest, "source-manifest.json")])
        return members

    def shared_inputs(self) -> None:
        """Pin reusable artifacts to the v5 objects, without duplicating weights."""
        paths = ["model/manifest.json", "encoder/train-embeddings.npz", "encoder/train-embeddings.json",
                 "encoder/test-embeddings.npz", "encoder/test-embeddings.json"]
        references = []
        for relative in paths:
            record = ArchiveFile.capture(self.v5 / relative, relative).record()
            record["local_artifact"] = str((self.v5 / relative).relative_to(self.root))
            record["archive_key"] = BASE_PREFIX + "improvement-v5/" + ("model-bundle.zip" if relative.startswith("model/") else "run-bundle.zip")
            references.append(record)
        write_json(self.run / "shared-inputs-manifest.json", {
            "bucket": BUCKET, "source_experiment": "improvement-v5",
            "model_archive_sha256": digest(self.v5 / "model-bundle.zip"),
            "model_archive_key": BASE_PREFIX + "improvement-v5/model-bundle.zip",
            "experiment_archive_sha256": digest(self.v5 / "run-bundle.zip"),
            "files": references,
        })

    def package(self) -> dict:
        has_model = (self.run / "model/manifest.json").is_file()
        if has_model:
            self.model()
        elif self.experiment == "improvement-v6":
            self.shared_inputs()
        else:
            raise ValueError("Experiment has neither its own pinned model nor defined shared inputs")
        source = self.source()
        paths = sorted(path for path in self.run.rglob("*")
                       if path.is_file() and path.name not in GENERATED
                       and not set(path.relative_to(self.run).parts) & EXCLUDED_PARTS
                       # Staging copies the model; retain sibling dataset records only.
                       and path.relative_to(self.run).parts[0] != "kaggle-dataset-upload"
                       and path.suffix not in {".zip", ".bundle", ".partial", ".tmp"})
        if any(path.suffix == ".safetensors" for path in paths):
            raise ValueError("Model weights belong only in model-bundle.zip")
        if not has_model:
            paths.append(self.run / "shared-inputs-manifest.json")
        members = [ArchiveFile.capture(path, str(path.relative_to(self.run))) for path in paths]
        manifest = self.run / "run-manifest.json"
        write_json(manifest, {"experiment": self.experiment, "files": [member.record() for member in members]})
        bundle(self.run / "run-bundle.zip", members + [ArchiveFile.capture(manifest, "run-manifest.json")])
        direct = {name: self.run / name for name in ("run-bundle.zip", "source.zip", "run-manifest.json", "source-manifest.json")}
        direct["README.md"] = self.root / "README.md"
        if has_model:
            direct.update({name: self.run / name for name in ("model-bundle.zip", "model-bundle-manifest.json")})
        else:
            direct["shared-inputs-manifest.json"] = self.run / "shared-inputs-manifest.json"
        for name in ("kaggle_submission.json", "checkpoint.bundle", "checkpoint.json"):
            if (self.run / name).exists():
                direct[name] = self.run / name
        notebook_dir = self.root / "notebooks" / self.experiment
        for path in sorted(notebook_dir.glob("*.ipynb")):
            if not path.name.endswith("executed.ipynb"):
                direct[path.name] = path
        archive_manifest = {"schema_version": 1, "bucket": BUCKET, "region": REGION,
                            "experiment": self.experiment, "objects": []}
        for name, path in direct.items():
            member = ArchiveFile.capture(path, name)
            archive_manifest["objects"].append({
                "local_path": str(path), "key": BASE_PREFIX + self.experiment + "/" + name,
                "size_bytes": member.size_bytes, "sha256": member.sha256,
            })
        write_json(self.run / "archive-manifest.json", archive_manifest)
        return {"experiment": self.experiment, "source_files": len(source), "experiment_files": len(members),
                "objects": len(direct), "total_bytes": sum(item["size_bytes"] for item in archive_manifest["objects"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("improvement-v5", "improvement-v6", "improvement-v7"), default="improvement-v5")
    parser.add_argument("--model-only", action="store_true", help="Prepare and verify only this experiment's immutable model bundle")
    args = parser.parse_args()
    packager = ExperimentPackager(Path(__file__).resolve().parents[1], args.experiment)
    if args.model_only:
        packager.model()
        print(json.dumps({"model_bundle_verified": str(packager.run / "model-bundle.zip")}))
    else:
        print(json.dumps(packager.package()))


if __name__ == "__main__":
    main()
