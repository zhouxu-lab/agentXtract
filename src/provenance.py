"""Write a privacy-safe, machine-readable provenance record for an assembly."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SENSITIVE_KEYS = {
    "api_key", "apikey", "password", "secret", "client_secret",
    "access_token", "refresh_token", "bearer_token", "authorization",
}
_DEPENDENCIES = (
    "anthropic",
    "docling",
    "duckdb",
    "google-genai",
    "numpy",
    "openai",
    "pandas",
    "pyarrow",
    "pydantic",
    "pymupdf",
    "scipy",
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tree_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _portable_config(value: Any, key: str = "") -> Any:
    """Remove secrets and normalize local project paths before serialization."""
    lowered = key.lower()
    if lowered in _SENSITIVE_KEYS or lowered.endswith("_api_key"):
        return "<redacted>"
    if lowered == "overrides_file":
        return "<local-assembly-override>" if value else value
    if isinstance(value, dict):
        return {
            str(child_key): _portable_config(child_value, str(child_key))
            for child_key, child_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_portable_config(item, key) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and lowered.endswith(("_dir", "_path")):
        path = Path(value)
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except (OSError, ValueError):
            return {
                "external_name": path.name,
                "path_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            }
    return value


def _git_state() -> dict[str, Any]:
    def _git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in _DEPENDENCIES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def write_run_provenance(
    output_dir: Path,
    dataframe,
    extraction_results: list,
    parsed_papers: dict,
    config: dict | None,
    assembly_extensions: dict[str, Any] | None = None,
) -> Path:
    """Atomically describe the inputs and software that produced an export."""
    portable_config = _portable_config(config or {})
    extraction_hashes = sorted(
        _canonical_sha256(result.model_dump(mode="json"))
        for result in extraction_results
    )
    paper_rows = sorted({
        (
            getattr(paper, "paper_uid", ""),
            paper.metadata.doi or paper.doi,
            _canonical_sha256(paper.model_dump(mode="json")),
        )
        for paper in parsed_papers.values()
    })
    source_files = list((PROJECT_ROOT / "src").rglob("*.py"))
    prompt_files = list((PROJECT_ROOT / "skills").rglob("*.md"))
    config_files = list((PROJECT_ROOT / "configs").rglob("*.yaml"))
    models = sorted(
        {
            str(model)
            for model in dataframe.get("extraction_model", [])
            if model is not None and str(model).strip()
        }
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(dataframe),
        "papers": [
            {"paper_uid": uid, "doi": doi, "parsed_paper_sha256": digest}
            for uid, doi, digest in paper_rows
        ],
        "models_recorded": models,
        "extraction_result_sha256": extraction_hashes,
        "runtime_config": portable_config,
        "runtime_config_sha256": _canonical_sha256(portable_config),
        "assembly_extensions": assembly_extensions or {},
        "assembly_extensions_sha256": _canonical_sha256(
            assembly_extensions or {}
        ),
        "source_tree_sha256": _tree_sha256(source_files),
        "prompt_tree_sha256": _tree_sha256(prompt_files),
        "config_tree_sha256": _tree_sha256(config_files),
        "git": _git_state(),
        "dependencies": _dependency_versions(),
        "primary_output_sha256": {
            name: hashlib.sha256((Path(output_dir) / name).read_bytes()).hexdigest()
            for name in (
                "dielectric_properties.csv",
                "dielectric_properties.parquet",
                "cowork.duckdb",
            )
            if (Path(output_dir) / name).is_file()
        },
    }

    output_path = Path(output_dir) / "run_provenance.json"
    temporary = output_path.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(output_path)
    return output_path
