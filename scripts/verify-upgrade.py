"""Train using a released source tag, then branch/compare/export with this backend."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from dsh_ml_workbench import __version__
from dsh_ml_workbench.files import atomic_json, sha256
from dsh_ml_workbench.service import Workbench
from sklearn.datasets import make_classification

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--from-version", default="0.1.0", choices=["0.1.0", "0.2.0", "0.3.0"])
parser.add_argument("--root", type=Path, default=root / ".verification" / "legacy-upgrade")
parser.add_argument("--report", type=Path, default=root / "docs" / "upgrade-results.json")
args = parser.parse_args()
temporary = args.root.resolve()
temporary.mkdir(parents=True, exist_ok=False)
package = temporary / "old-python" / "dsh_ml_workbench"
package.mkdir(parents=True)
git = ["git", "-c", f"safe.directory={root.as_posix()}"]
tag = "v" + args.from_version
names = subprocess.check_output([*git, "ls-tree", "-r", "--name-only", tag, "python/dsh_ml_workbench/"], cwd=root).decode().splitlines()
for name in names:
    if name.endswith(".py"):
        (package / Path(name).name).write_bytes(subprocess.check_output([*git, "show", f"{tag}:{name}"], cwd=root))
x, y = make_classification(n_samples=100, n_features=4, random_state=31)
data = pd.DataFrame(x, columns=["a", "b", "c", "d"])
data["target"] = y
csv = temporary / "data.csv"
data.to_csv(csv, index=False)
records = temporary / "records"
environment = dict(os.environ, PYTHONPATH=str(package.parent), PYTHONUTF8="1")


def old_call(operation, args):
    request = {"protocol_version": 1, "operation": operation, "args": args}
    result = subprocess.run([sys.executable, "-m", "dsh_ml_workbench", "rpc", "--root", str(records)],
        input=json.dumps(request), capture_output=True, text=True, env=environment, timeout=120)
    response = json.loads(result.stdout)
    assert response["ok"], response
    return response["result"]


old = old_call("create", {"dataset_path": str(csv), "task": "classification", "target": "target", "model": "logistic_regression"})
old_call("start", {"run_id": old["run_id"]})
workbench = Workbench(records)
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    completed = workbench.get(old["run_id"])
    if completed["state"] in {"succeeded", "failed", "interrupted"}:
        break
    time.sleep(0.2)
assert completed["state"] == "succeeded", completed
branch = workbench.branch(old["run_id"], {"model": "random_forest", "params": {"n_estimators": 10}})
assert sha256(workbench.run_dir(old["run_id"]) / "inputs/engine.py") == sha256(workbench.run_dir(branch["run_id"]) / "inputs/engine.py")
workbench.start(branch["run_id"])
deadline = time.monotonic() + 180
while time.monotonic() < deadline:
    completed = workbench.get(branch["run_id"])
    if completed["state"] in {"succeeded", "failed", "interrupted"}:
        break
    time.sleep(0.2)
assert completed["state"] == "succeeded", completed
comparison = workbench.compare([old["run_id"], branch["run_id"]])
workbench.export(branch["run_id"])
atomic_json(args.report, {"from": args.from_version, "to": __version__,
    "legacy_fit": "passed", "branch_with_original_recipe": "passed", "comparison": comparison,
    "legacy_export": "passed"})
print("Legacy source training, fitted-preprocessor branch, comparison and export passed")
