"""Atomic local snapshots, hashes, and portable environment evidence."""

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

from .spec import WorkbenchError


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def tree_hashes(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise WorkbenchError("UNSAFE_ARTIFACT", "Snapshot artifacts must not contain symbolic links")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256(path)
    return result


def verify_tree(root: Path, expected: dict[str, str]) -> None:
    actual = tree_hashes(root)
    if actual != expected:
        changed = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
        raise WorkbenchError("INTEGRITY_MISMATCH", "Snapshot files changed: " + ", ".join(changed[:8]))


def environment() -> dict[str, Any]:
    names = ["mlflow", "scikit-learn", "pandas", "numpy", "scipy", "cloudpickle", "joblib"]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in names},
    }


def requirements() -> str:
    # Pin installed distributions, excluding this locally installed adapter package.
    packages = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name") and dist.metadata["Name"].lower().replace("_", "-") != "dsh-ml-workbench"
    }
    return "".join(f"{name}=={version}" + ('; platform_system == "Windows"' if name.lower() == "pywin32" else '') + "\n"
                   for name, version in sorted(packages.items(), key=lambda x: x[0].lower()))
