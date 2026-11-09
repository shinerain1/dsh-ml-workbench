"""Standalone export verifier and retrainer; requires no workbench installation."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
from pathlib import Path


def verify(folder: Path) -> dict:
    manifest = json.loads((folder / "checksums.json").read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe artifact: {relative}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Artifact hash mismatch: {relative}")
    return {"verified_files": len(manifest)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reproduced"))
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    status = verify(folder)
    if args.verify_only:
        print(json.dumps(status))
        return 0
    expected_env = json.loads((folder / "inputs" / "environment.json").read_text(encoding="utf-8"))
    different = [name for name, version in expected_env["packages"].items()
                 if importlib.metadata.version(name) != version]
    if different:
        raise ValueError("Install inputs/requirements.txt first; differing packages: " + ", ".join(different))
    data = args.data.resolve() if args.data else folder / "inputs" / "dataset.csv"
    if not data.is_file():
        raise ValueError("Data is not included. Pass --data ORIGINAL.csv; its SHA-256 must match")
    command = [sys.executable, "-B", str(folder / "inputs" / "engine.py"), "--inputs", str(folder / "inputs"),
               "--output", str(args.output.resolve()), "--data", str(data)]
    subprocess.run(command, check=True)
    recorded = json.loads((folder / "output" / "result.json").read_text(encoding="utf-8"))["metrics"]
    repeated = json.loads((args.output / "result.json").read_text(encoding="utf-8"))["metrics"]
    matches = recorded.keys() == repeated.keys() and all(
        math.isclose(recorded[key], repeated[key], rel_tol=1e-7, abs_tol=1e-8) for key in recorded)
    print(json.dumps({**status, "metrics_match": matches, "recorded": recorded, "reproduced": repeated}))
    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
