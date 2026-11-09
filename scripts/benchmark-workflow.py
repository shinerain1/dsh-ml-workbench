"""Controlled mechanism comparison, not a natural-language DSH Agent A/B trial.

The no-adapter arm receives the same frozen sklearn recipe and inputs (a generous
baseline), logs them with native MlflowClient, and runs them without Workbench.
Positive controls add explicit MLflow tags/checks to demonstrate that a careful
script can implement the same safeguards. No human-intervention metric is inferred.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MLFLOW_ENABLE_TELEMETRY"] = "false"

import pandas as pd
from dsh_ml_workbench.__main__ import dispatch
from dsh_ml_workbench.engine import train
from dsh_ml_workbench.files import sha256
from dsh_ml_workbench.service import Workbench
from dsh_ml_workbench.spec import ExperimentSpec, RpcRequest, WorkbenchError
from mlflow import MlflowClient
from sklearn.datasets import make_classification, make_regression


def wait(workbench, run_id):
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        result = workbench.get(run_id)
        if result["state"] == "succeeded":
            return result
        if result["state"] in {"failed", "interrupted", "cancelled"}:
            raise AssertionError(result)
        time.sleep(0.2)
    workbench.cancel(run_id)
    raise TimeoutError(run_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reproduce-python", default=sys.executable)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    workbench = Workbench(root / "plugin")
    baseline = MlflowClient(tracking_uri="sqlite:///" + (root / "baseline.sqlite3").as_posix())
    experiment = baseline.create_experiment("direct-mlflow", artifact_location=(root / "baseline-artifacts").as_uri())
    rows = []
    pairs = []
    for task, seed in [("classification", 7), ("classification", 19), ("classification", 41), ("regression", 7)]:
        x, y = (make_classification(n_samples=160, n_features=6, n_informative=4, random_state=7)
                if task == "classification" else make_regression(n_samples=160, n_features=6, noise=2, random_state=7))
        frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(6)])
        frame["target"] = y
        data = root / f"{task}-{seed}.csv"
        frame.to_csv(data, index=False)
        spec = ExperimentSpec(dataset_path=str(data), target="target", task=task, seed=seed,
                              model="logistic_regression" if task == "classification" else "ridge")
        created = workbench.create(spec)
        inputs = workbench.run_dir(created["run_id"]) / "inputs"
        # Both arms get exactly these immutable inputs; baseline uses no workflow methods.
        plain = baseline.create_run(experiment, tags={"split_sha256": sha256(inputs / "split.json"),
                                                     "dataset_sha256": sha256(inputs / "dataset.csv")}).info.run_id
        for key, value in {"seed": seed, "model": spec.model}.items():
            baseline.log_param(plain, key, value)
        baseline.log_artifacts(plain, str(inputs), "inputs")
        direct_output = root / f"direct-{task}-{seed}"
        direct = train(inputs, direct_output)
        for key, value in direct["metrics"].items():
            baseline.log_metric(plain, key, value)
        baseline.log_artifacts(plain, str(direct_output), "output")
        baseline.set_terminated(plain)
        workbench.start(created["run_id"])
        managed = wait(workbench, created["run_id"])
        delta = max(abs(direct["metrics"][m] - managed["metrics"][m]) for m in direct["metrics"])
        assert delta < 1e-12, (direct, managed)
        baseline_inputs = Path(url2pathname(urlsplit(baseline.get_run(plain).info.artifact_uri).path)) / "inputs"
        repeated = train(baseline_inputs, root / f"direct-reproduced-{task}-{seed}")
        baseline_reproduced = repeated["metrics"] == direct["metrics"]
        delivery = workbench.export(created["run_id"], include_data=True)
        unpacked = root / f"delivery-{task}-{seed}"
        with zipfile.ZipFile(delivery["path"]) as archive:
            archive.extractall(unpacked)
        check = subprocess.run([args.reproduce_python, "reproduce.py"], cwd=unpacked,
                               capture_output=True, text=True, timeout=150)
        assert check.returncode == 0, check.stdout + check.stderr
        plugin_reproduced = json.loads(check.stdout.splitlines()[-1])["metrics_match"]
        required = ["dataset.csv", "config.json", "split.json", "engine.py", "environment.json", "requirements.txt"]
        # Six input categories plus native metrics and fitted model = eight record categories.
        baseline_complete = sum((baseline_inputs / f).is_file() for f in required) + bool(baseline.get_run(plain).data.metrics) + (baseline_inputs.parent / "output/model/model.pkl").is_file()
        plugin_complete = sum((inputs / f).is_file() for f in required) + bool(managed["metrics"]) + (inputs.parent / "output/model/model.pkl").is_file()
        assert baseline_reproduced and plugin_reproduced and baseline_complete == plugin_complete == 8
        rows.append({"task": task, "seed": seed, "direct_mlflow_metrics": direct["metrics"],
                     "plugin_metrics": managed["metrics"], "maximum_absolute_metric_delta": delta,
                     "direct_mlflow_record_categories": baseline_complete, "plugin_record_categories": plugin_complete,
                     "record_categories_total": 8, "direct_mlflow_reproduced": baseline_reproduced,
                     "plugin_standalone_reproduced": plugin_reproduced})
        pairs.append((plain, created["run_id"]))
        print(json.dumps({"completed_pair": len(rows), "task": task, "seed": seed}), flush=True)

    # Same data, different validation rows: raw MLflow permits sorting stored scores.
    ids = [pairs[i][0] for i in (0, 1)]
    raw_ranking = baseline.search_runs([experiment], filter_string="attributes.run_id IN ('" + "','".join(ids) + "')",
                                       order_by=["metrics.accuracy DESC"])
    assert len(raw_ranking) == 2
    explicit_reject = len({baseline.get_run(r).data.tags["split_sha256"] for r in ids}) != 1
    try:
        workbench.compare([pairs[i][1] for i in (0, 1)])
        raise AssertionError("incompatible ranking was accepted")
    except WorkbenchError as error:
        assert error.code == "INCOMPARABLE_EXPERIMENTS"

    # Mutate both artifact stores after success. Native get_run doesn't enforce file hashes.
    plain, managed_id = pairs[0]
    plain_file = Path(url2pathname(urlsplit(baseline.get_run(plain).info.artifact_uri).path)) / "output/model/model.pkl"
    expected = sha256(plain_file)
    original = plain_file.read_bytes()
    plain_file.write_bytes(original + b"injected-mutation")
    raw_accepts = baseline.get_run(plain).info.status == "FINISHED"
    checked_rejects = sha256(plain_file) != expected
    plain_file.write_bytes(original)
    managed_file = workbench.run_dir(managed_id) / "output/model/model.pkl"
    original = managed_file.read_bytes()
    managed_file.write_bytes(original + b"injected-mutation")
    try:
        workbench.export(managed_id)
        raise AssertionError("mutated model was exported")
    except WorkbenchError as error:
        assert error.code == "INTEGRITY_MISMATCH"
    finally:
        managed_file.write_bytes(original)

    # Lost creation response: duplicate native calls vs explicit tag search vs Session identity.
    raw_ids = [baseline.create_run(experiment).info.run_id for _ in range(2)]
    manual = baseline.create_run(experiment, tags={"request_key": "manual-intent"}).info.run_id
    recovered = baseline.search_runs([experiment], filter_string="tags.request_key = 'manual-intent'")[0].info.run_id
    session = Workbench(workbench.root, dsh_context={"session_id": "benchmark-session", "call_id": "create-lost"})
    request = RpcRequest(operation="create", args=spec.model_dump())
    plugin_ids = [dispatch(session, request)["run_id"], dispatch(Workbench(workbench.root, dsh_context=session.dsh_context), request)["run_id"]]
    assert len(set(raw_ids)) == 2 and recovered == manual and len(set(plugin_ids)) == 1
    assert explicit_reject and raw_accepts and checked_rejects
    report = {"kind": "deterministic-workflow-mechanism-comparison", "model_calls": 0,
              "agent_ab_test": False, "human_interventions_measured": False,
              "baseline": "Direct MLflow with the same frozen sklearn recipe, full input/model logging, and separate explicit-check positive controls",
              "pairs": rows, "faults": {
                  "lost_creation_response_unique_runs": {"raw_mlflow": 2, "mlflow_with_explicit_tag_lookup": 1, "plugin_same_dsh_call": 1},
                  "incompatible_split_ranking_rejected": {"raw_mlflow": False, "mlflow_with_explicit_split_check": explicit_reject, "plugin": True},
                  "post_success_model_mutation_rejected": {"raw_mlflow_get_run": not raw_accepts, "mlflow_with_explicit_hash_check": checked_rejects, "plugin_export": True}},
              "limitations": ["Shared adapter recipe isolates workflow behavior; it is not independently authored DSH code.",
                              "Explicit baseline tag lookup is serial, not a concurrency-safe idempotency service.",
                              "No LLM, human-intervention, cost, speed or model-quality advantage is measured.",
                              "Fault cases are preselected design checks, not estimates of real incident frequency."]}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs_passed": len(rows), "report": str(args.report)}), flush=True)


if __name__ == "__main__":
    main()
