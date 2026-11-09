"""Run real synthetic-data experiments and save a recomputable delivery report."""

import argparse
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
from dsh_ml_workbench.files import atomic_json, environment, sha256
from dsh_ml_workbench.service import Workbench
from dsh_ml_workbench.spec import ExperimentSpec, WorkbenchError
from sklearn.datasets import make_classification, make_regression


def finished(workbench, run):
    workbench.start(run["run_id"])
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        result = workbench.get(run["run_id"])
        if result["state"] == "succeeded":
            return result
        if result["state"] in {"failed", "cancelled", "interrupted"}:
            raise RuntimeError(result)
        time.sleep(0.2)
    workbench.cancel(run["run_id"])
    raise TimeoutError(run["run_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if (args.root / "records").exists():
        raise ValueError("Use a fresh demo root so evidence cannot mix runs")
    workbench = Workbench(args.root / "records", allow_custom_code=True)
    x, y = make_classification(n_samples=600, n_features=8, n_informative=5, random_state=2026)
    data = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(8)])
    data["channel"] = ["online" if i % 2 else "store" for i in range(len(data))]
    data["repurchased"] = y
    csv = args.root / "synthetic_customers.csv"
    data.to_csv(csv, index=False)
    first = finished(workbench, workbench.create(ExperimentSpec(
        dataset_path=str(csv.resolve()), target="repurchased", task="classification", name="forest",
        model="random_forest", params={"n_estimators": 60})))
    second = finished(workbench, workbench.branch(first["run_id"], {"name": "extra trees", "model": "extra_trees", "params": {"n_estimators": 60}}))
    third = finished(workbench, workbench.branch(first["run_id"], {"name": "robust preprocessing", "preprocessor_path": str(Path('examples/preprocessor.py').resolve())}, reuse_preprocessing=False))
    fourth = finished(workbench, workbench.branch(third["run_id"], {"name": "third preprocessing, linear model", "model": "logistic_regression", "params": {}}))
    classification = workbench.compare([r["run_id"] for r in (first, second, third, fourth)])
    x, y = make_regression(n_samples=400, n_features=6, noise=5, random_state=2026)
    regression_data = pd.DataFrame(x, columns=[f"x{i}" for i in range(6)])
    regression_data["target"] = y
    regression_csv = args.root / "synthetic_regression.csv"
    regression_data.to_csv(regression_csv, index=False)
    reg1 = finished(workbench, workbench.create(ExperimentSpec(dataset_path=str(regression_csv.resolve()), target="target", task="regression", model="ridge", name="ridge")))
    reg2 = finished(workbench, workbench.branch(reg1["run_id"], {"name": "regression forest", "model": "random_forest", "params": {"n_estimators": 40}}))
    regression = workbench.compare([reg1["run_id"], reg2["run_id"]])
    deliveries = []
    for run in [fourth, reg1]:
        delivery = workbench.export(run["run_id"], include_data=True)
        directory = args.root / "deliveries" / run["run_id"]
        with zipfile.ZipFile(delivery["path"]) as archive:
            archive.extractall(directory)
        process = subprocess.run([sys.executable, "reproduce.py"], cwd=directory, capture_output=True, text=True, timeout=180)
        if process.returncode:
            raise RuntimeError(process.stdout + process.stderr)
        deliveries.append({"run_id": run["run_id"], "sha256": sha256(Path(delivery["path"])),
                           "verification": json.loads(process.stdout.splitlines()[-1])})
    incompatible = finished(workbench, workbench.branch(first["run_id"], {"name": "different split", "seed": 91}, reuse_preprocessing=False))
    try:
        workbench.compare([first["run_id"], incompatible["run_id"]])
    except WorkbenchError as error:
        refusal = error.code
    else:
        raise AssertionError("Different validation rows were incorrectly compared")
    report = {"format": 1, "data": "synthetic, generated with sklearn; no customer records",
              "environment": environment(), "classification": classification, "regression": regression,
              "deliveries": deliveries, "incompatible_comparison": refusal,
              "native_dsh_agent_comparison": "not measured by this deterministic integration demo",
              "manual_intervention_rate": None}
    atomic_json(args.report, report)
    print(json.dumps({"report": str(args.report), "successful_runs": 7, "reproduced_deliveries": len(deliveries)}))


if __name__ == "__main__":
    main()
