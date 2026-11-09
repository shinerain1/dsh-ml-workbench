"""Check release contents and reproduce both synthetic deliveries in a fresh interpreter."""

import argparse
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--demo-root", type=Path, required=True)
    args = parser.parse_args()
    interpreter = args.python.resolve()
    root = Path(__file__).resolve().parents[1]
    with tarfile.open(root / "dist" / "dsh-ml-workbench-0.1.0.tgz") as archive:
        names = archive.getnames()
    assert "package/lib/index.js" in names and "package/cordis.patch.yml" in names
    assert not any("__pycache__" in name or ".egg-info" in name or ".verification" in name for name in names)
    code = "import dsh_ml_workbench,json; print(json.dumps({'version':dsh_ml_workbench.__version__,'installed':dsh_ml_workbench.__file__}))"
    check = subprocess.run([str(interpreter), "-c", code], capture_output=True, text=True, check=True)
    installed = json.loads(check.stdout)
    assert "site-packages" in installed["installed"], installed
    results = []
    for archive_path in sorted((args.demo_root / "records" / "exports").glob("*.zip")):
        destination = root / ".verification" / "clean-delivery" / archive_path.stem
        if destination.exists():
            raise ValueError(f"Use a fresh verification directory: {destination}")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        check = subprocess.run([str(interpreter), "reproduce.py"], cwd=destination,
                               capture_output=True, text=True, timeout=180)
        if check.returncode:
            raise RuntimeError(check.stdout + check.stderr)
        results.append(json.loads(check.stdout.splitlines()[-1]))
    assert len(results) == 2
    report = {"wheel_install": "passed", "backend_version": installed["version"],
              "npm_content_check": "passed", "clean_environment_deliveries": results}
    (root / "docs" / "packaging-results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
