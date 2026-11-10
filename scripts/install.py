"""Create an isolated backend and a reviewable DSH config snippet; never overwrite a DSH profile."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, help="Matching release wheel; defaults to installing this source package")
    parser.add_argument("--execution-mode", choices=["dsh", "local"], default="dsh")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    version = json.loads((source / "package.json").read_text(encoding="utf-8"))["version"]
    target = args.venv.resolve()
    if target.exists() and not (target / "pyvenv.cfg").is_file():
        parser.error("--venv exists and is not a Python virtual environment; choose an empty/new path")
    if not target.exists():
        subprocess.run([sys.executable, "-m", "venv", str(target)], check=True)
    python = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    package = args.wheel.resolve() if args.wheel else source
    subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r",
                    str(source / "requirements.lock"), str(package)], check=True)
    installed = subprocess.check_output([str(python), "-c", "from dsh_ml_workbench import __version__; print(__version__)"], text=True).strip()
    if installed != version:
        raise ValueError(f"Wheel version {installed} differs from plugin {version}; use matching release assets")
    root = args.workspace.resolve() / ".dsh-ml-workbench"
    subprocess.run([str(python), "-m", "dsh_ml_workbench", "doctor", "--root", str(root)], check=True)
    snippet = {"pythonExecutable": str(python), "root": str(root), "executionMode": args.execution_mode,
               "allowCustomCode": False, "requestTimeoutMs": 120000, "maxResponseBytes": 4194304}
    output = target / "ml-workbench.config.json"
    output.write_text(json.dumps(snippet, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"version": version, "config_snippet": str(output), "config": snippet,
                      "next": "Apply this config to the installed ml-workbench plugin, then call ml_workbench_health inside DSH."}))


if __name__ == "__main__":
    main()
