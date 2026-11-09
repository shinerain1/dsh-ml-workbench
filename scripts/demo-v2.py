"""Exercise real group/time experiments, persistent Optuna tuning and batch prediction."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from demo import finished
from dsh_ml_workbench.files import atomic_json, read_json
from dsh_ml_workbench.service import Workbench
from dsh_ml_workbench.spec import ExperimentSpec, SplitSpec
from sklearn.datasets import make_classification, make_regression

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
if args.root.exists():
    raise ValueError("Use a fresh root")
args.root.mkdir(parents=True)
workbench = Workbench(args.root / "records")
x, y = make_classification(n_samples=360, n_features=6, n_informative=4, random_state=88)
frame = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(x.shape[1])])
frame["customer_id"], frame["repurchased"] = np.arange(len(frame)) // 3, y
csv = args.root / "synthetic_grouped.csv"
frame.to_csv(csv, index=False)
base = finished(workbench, workbench.create(ExperimentSpec(dataset_path=str(csv.resolve()), target="repurchased",
    task="classification", name="grouped customer holdout", model="random_forest", params={"n_estimators": 20},
    split=SplitSpec(strategy="group", column="customer_id"))))
indices = read_json(workbench.run_dir(base["run_id"]) / "inputs" / "split.json")
assert set(frame.customer_id.iloc[indices["train"]]).isdisjoint(frame.customer_id.iloc[indices["test"]])
workbench.search_create(study_name="forest-depth", run_id=base["run_id"], n_trials=3, metric="roc_auc",
    reuse_preprocessing=True, search_space={"max_depth": {"type": "int", "low": 2, "high": 6}})
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    # Deliberately reconstruct the controller each step, as separate DSH calls do.
    search = Workbench(workbench.root).search_step("forest-depth")
    if search["state"] == "complete":
        break
    time.sleep(0.3)
else:
    workbench.search_cancel("forest-depth")
    raise TimeoutError("Search did not complete")
assert len(search["trials"]) == 3 and all(t["state"] == "COMPLETE" for t in search["trials"])
best = search["best"]["run_id"]
unlabeled = args.root / "unlabeled.csv"
frame.drop(columns=["repurchased", "customer_id"]).iloc[:20].to_csv(unlabeled, index=False)
prediction = workbench.predict(best, str(unlabeled.resolve()))
assert prediction["rows"] == 20
x, y = make_regression(n_samples=240, n_features=4, noise=3, random_state=89)
frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
frame["date"] = np.repeat(pd.date_range("2025-01-01", periods=80).astype(str), 3)
frame["target"] = y
csv = args.root / "synthetic_chronological.csv"
frame.to_csv(csv, index=False)
chronological = finished(workbench, workbench.create(ExperimentSpec(dataset_path=str(csv.resolve()),
    target="target", task="regression", name="chronological holdout", model="ridge",
    split=SplitSpec(strategy="time", column="date"))))
indices = read_json(workbench.run_dir(chronological["run_id"]) / "inputs" / "split.json")
assert max(frame.date.iloc[indices["train"]]) < min(frame.date.iloc[indices["test"]])
for run_id in [best, chronological["run_id"]]:
    workbench.export(run_id, include_data=True)
search.pop("storage_uri")
prediction.pop("path")
atomic_json(args.report, {"version": "0.2.0", "data": "synthetic, not customer records", "group_isolation": True,
    "chronological_isolation": True, "search": search, "prediction": prediction,
    "chronological_metrics": chronological["metrics"], "chronological_baseline": chronological["baseline_metrics"],
    "best_diff": workbench.diff(base["run_id"], best)})
print({"report": str(args.report), "optuna_trials": 3, "predicted_rows": 20, "exports": 2})
