"""Installation diagnostics without loading credentials or changing DSH configuration."""

import importlib.metadata
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path

from . import __version__


def diagnose(root: Path) -> dict:
    checks = []
    def check(name, action):
        try:
            detail = action()
            checks.append({"name": name, "ok": True, "detail": detail})
        except Exception as error:
            checks.append({"name": name, "ok": False, "detail": f"{type(error).__name__}: {error}"})

    def dependencies():
        from packaging.requirements import Requirement
        results = {}
        for value in importlib.metadata.requires("dsh-ml-workbench") or []:
            requirement = Requirement(value)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            version = importlib.metadata.version(requirement.name)
            if version not in requirement.specifier:
                raise ValueError(f"{requirement.name} {version} does not satisfy {requirement.specifier}")
            results[requirement.name] = version
        if importlib.metadata.version("dsh-ml-workbench") != __version__:
            raise ValueError("Imported source and installed distribution versions disagree; reinstall the matching wheel")
        return results

    def storage():
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="doctor-", dir=root) as folder:
            database = Path(folder) / "check.sqlite3"
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute("create table probe (value integer)")
                connection.execute("insert into probe values (1)")
            database.rename(Path(folder) / "renamed.sqlite3")
        return "SQLite write, commit and atomic rename succeeded"

    def node():
        executable = shutil.which("node")
        if not executable:
            raise ValueError("Node executable is not on PATH; DSH requires Node >=24.9")
        version = subprocess.check_output([executable, "--version"], text=True, timeout=10).strip()
        if tuple(map(int, version.lstrip("v").split(".")[:2])) < (24, 9):
            raise ValueError("Node >=24.9 required, found " + version)
        return version

    check("backend_dependencies", dependencies)
    check("storage", storage)
    check("node", node)
    return {"ok": all(item["ok"] for item in checks), "backend_version": __version__,
            "python_executable": sys.executable, "python_version": sys.version.split()[0],
            "isolated_environment": sys.prefix != sys.base_prefix, "root": str(root.resolve()),
            "checks": checks, "notes": ["This probes local installation only; DSH sandbox permissions are checked by an actual DSH tool call.",
                                        "The validated numerical environment is requirements.lock; other satisfying versions are not a reproduction guarantee."]}
