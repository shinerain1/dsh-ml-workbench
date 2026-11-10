"""Prespecified real-data mechanism experiment; no LLM or productivity claim."""
# ruff: noqa: E402
import argparse
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MLFLOW_ENABLE_TELEMETRY"] = "false"

import pandas as pd
from dsh_ml_workbench.engine import evaluate_final, train
from dsh_ml_workbench.files import environment, read_json, sha256, tree_hashes
from dsh_ml_workbench.service import Workbench
from dsh_ml_workbench.spec import ExperimentSpec, Preprocessing, SplitSpec, WorkbenchError
from mlflow import MlflowClient


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def wait(workbench, run_id):
    until = time.monotonic() + 600
    while time.monotonic() < until:
        record = workbench.wait(run_id, wait_seconds=5)
        if record["state"] == "succeeded":
            return record
        if record["state"] in {"failed", "cancelled", "interrupted"}:
            raise RuntimeError(json.dumps(record))
    workbench.cancel(run_id)
    raise TimeoutError(run_id)


def rejects(action, code):
    try:
        action()
    except WorkbenchError as error:
        return {"passed": error.code == code, "code": error.code, "message": str(error)}
    return {"passed": False, "code": "ACCEPTED"}


def copied_reproduction(python, folder, command):
    result = subprocess.run([python, *command], cwd=folder, capture_output=True,
                            text=True, encoding="utf-8", timeout=600,
                            env={**os.environ, "PYTHONUTF8": "1"})
    (folder / "execution.stdout").write_text(result.stdout, encoding="utf-8")
    (folder / "execution.stderr").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:] + result.stdout[-2000:])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reproduce-python", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    reproduce_python = str(Path(args.reproduce_python).resolve())
    data_root = args.data_root.resolve()
    report = {"kind": "real-data-deterministic-workflow-comparison", "model_calls": 0,
              "agent_ab_test": False, "environment": environment(),
              "protocol_sha256": sha256(Path(__file__).resolve().parents[1] / "docs/REAL-DATA-PROTOCOL.md"),
              "provenance": read_json(data_root / "provenance.json"), "datasets": [],
              "limitations": ["Shared frozen recipe isolates workflow behavior, not independently authored Agent code.",
                              "Injected checks do not estimate real-world error frequency.",
                              "Single-machine sequential timings include cache/order effects; no speed claim.",
                              "Final holdout is used once per dataset; no confidence interval or external validation.",
                              "Direct MLflow with explicit checks can implement the same protections."]}
    for name, task, time_column in [("retail", "classification", "anchor"), ("bike", "regression", "timestamp")]:
        folder = root / name
        folder.mkdir()
        session_id = "real-data-" + name
        workbench = Workbench(folder / "plugin", dsh_context={"session_id": session_id, "call_id": "benchmark"})
        baseline = MlflowClient(tracking_uri="sqlite:///" + (folder / "direct.sqlite3").as_posix())
        experiment = baseline.create_experiment("direct-mlflow", artifact_location=(folder / "direct-artifacts").as_uri())
        frame = pd.read_csv(data_root / (name + ".csv"))
        times = pd.to_datetime(frame[time_column])
        metric = "average_precision" if task == "classification" else "rmse"
        linear = "logistic_regression" if task == "classification" else "ridge"
        candidates = [(linear, {"max_iter": 1000} if task == "classification" else {"alpha": 1.0}),
                      ("random_forest", {"n_estimators": 64, "min_samples_leaf": 5}),
                      ("hist_gradient_boosting", {"max_iter": 80, "max_leaf_nodes": 15})]
        dataset_report = {"name": name, "task": task, "selection_metric": metric, "pairs": [], "branch_checks": []}
        native_ids, outputs, parents = {}, {}, []
        specs = {}
        for seed in (7, 19, 41):
            parent = None
            for model, params in candidates:
                label = f"{name}-{seed}-{model}"
                before = time.perf_counter()
                if parent is None:
                    spec = ExperimentSpec(dataset_path=str(data_root / (name + ".csv")), target="target",
                                          task=task, name=label, model=model, params=params, seed=seed,
                                          split=SplitSpec(strategy="time", column=time_column),
                                          preprocessing=Preprocessing(drop_columns=["customer_id"] if name == "retail" else []))
                    created = workbench.create(spec, request_key=label)
                    specs[label] = spec
                else:
                    parent_dir = workbench.run_dir(parent)
                    parent_before = tree_hashes(parent_dir / "output")
                    created = workbench.branch(parent, changes={"name": label, "model": model, "params": params}, request_key=label)
                run_id = created["run_id"]
                inputs = workbench.run_dir(run_id) / "inputs"
                split = read_json(inputs / "split.json")
                if not dataset_report.get("partitions"):
                    dataset_report["partitions"] = {key: {"rows": len(rows), "start": times.iloc[rows].min().isoformat(),
                                                           "end": times.iloc[rows].max().isoformat(),
                                                           "target_mean": float(frame.target.iloc[rows].mean())}
                                                    for key, rows in split.items()}
                    assert times.iloc[split["train"]].max() < times.iloc[split["test"]].min() < times.iloc[split["final_test"]].min()
                    if name == "retail":
                        for left, right in [("train", "test"), ("test", "final_test")]:
                            assert times.iloc[split[left]].max() + pd.offsets.MonthBegin(1) <= times.iloc[split[right]].min()
                    dataset_report["split_sha256"] = sha256(inputs / "split.json")
                assert sha256(inputs / "split.json") == dataset_report["split_sha256"]
                # Start and complete the real background supervisor/worker path.
                workbench.start(run_id)
                managed = wait(workbench, run_id)
                managed_wall = time.perf_counter() - before
                result = read_json(workbench.run_dir(run_id) / "output/result.json")
                if parent is not None:
                    check = {"parent_run_id": parent, "child_run_id": run_id,
                             "parent_outputs_unchanged": parent_before == tree_hashes(parent_dir / "output"),
                             "fitted_preprocessor_identical": sha256(parent_dir / "output/preprocessor.pkl") == sha256(inputs / "reused_preprocessor.pkl")}
                    dataset_report["branch_checks"].append(check)
                    assert check["parent_outputs_unchanged"] and check["fitted_preprocessor_identical"]
                else:
                    parent = run_id
                    parents.append(run_id)

                # Native MLflow arm: identical inputs, its own fit/model/artifact logging.
                before = time.perf_counter()
                native = baseline.create_run(experiment, tags={"split_sha256": sha256(inputs / "split.json"),
                                                               "dataset_sha256": sha256(inputs / "dataset.csv"), "request_key": label}).info.run_id
                for key, value in {"seed": seed, "model": model, **params}.items():
                    baseline.log_param(native, key, value)
                baseline.log_artifacts(native, str(inputs), "inputs")
                direct_dir = folder / ("direct-" + label)
                direct = train(inputs, direct_dir)
                for key, value in direct["metrics"].items():
                    baseline.log_metric(native, key, value)
                baseline.log_artifacts(native, str(direct_dir), "output")
                baseline.set_terminated(native)
                delta = max(abs(value - managed["metrics"][key]) for key, value in direct["metrics"].items())
                assert delta < 1e-12
                native_ids[run_id] = native
                outputs[run_id] = direct_dir
                row = {"run_id": run_id, "native_run_id": native, "seed": seed, "model": model,
                       "metrics": managed["metrics"], "dummy_metrics": result["baseline_metrics"],
                       "maximum_absolute_metric_delta": delta, "plugin_fit_seconds": result["train_seconds"],
                       "direct_fit_seconds": direct["train_seconds"], "plugin_workflow_seconds": managed_wall,
                       "direct_workflow_seconds": time.perf_counter() - before}
                required = ["dataset.csv", "config.json", "split.json", "engine.py", "environment.json", "requirements.txt"]
                row["plugin_record_categories"] = sum((inputs / f).is_file() for f in required) + bool(managed["metrics"]) + (inputs.parent / "output/model/model.pkl").is_file()
                # Inspect the actually logged baseline artifacts, not the source inputs.
                from urllib.parse import urlsplit
                from urllib.request import url2pathname
                logged = Path(url2pathname(urlsplit(baseline.get_run(native).info.artifact_uri).path))
                row["direct_record_categories"] = sum((logged / "inputs" / f).is_file() for f in required) + bool(baseline.get_run(native).data.metrics) + (logged / "output/model/model.pkl").is_file()
                assert row["plugin_record_categories"] == row["direct_record_categories"] == 8
                dataset_report["pairs"].append(row)
                save(folder / "progress.json", dataset_report)
                print(json.dumps({"completed_pair": label, "validation_metric": managed["metrics"][metric]}), flush=True)

        all_ids = [r["run_id"] for r in dataset_report["pairs"]]
        comparison = workbench.compare(all_ids, metric=metric)
        selected = comparison["ranking"][0]["run_id"]
        dataset_report["selected_run_id"] = selected
        final = workbench.finalize(selected)["final_test"]
        dataset_report["final_test"] = final
        inputs = workbench.run_dir(selected) / "inputs"
        direct_final = evaluate_final(inputs, outputs[selected] / "model", folder / "direct-final")
        assert final["metrics"] == direct_final["metrics"]
        for key, value in direct_final["metrics"].items():
            baseline.log_metric(native_ids[selected], "final_test." + key, value)
        baseline.set_experiment_tag(experiment, "final_selected_run", native_ids[selected])

        # Separate random-split diagnostic, explicitly excluding identifiers/time predictors.
        random_spec = ExperimentSpec(dataset_path=str(data_root / (name + ".csv")), target="target", task=task,
                                    model="random_forest", params=candidates[1][1], seed=7,
                                    preprocessing=Preprocessing(drop_columns=[time_column] + (["customer_id"] if name == "retail" else [])))
        random_run = workbench.create(random_spec, request_key="random-diagnostic")
        workbench.start(random_run["run_id"])
        random_result = wait(workbench, random_run["run_id"])
        dataset_report["random_split_diagnostic"] = {"run_id": random_run["run_id"], "validation_metrics": random_result["metrics"],
                                                       "final_test_evaluated": False}
        faults = {"mixed_split_comparison": rejects(lambda: workbench.compare([parents[0], random_run["run_id"]], metric), "INCOMPARABLE_EXPERIMENTS"),
                  "second_final_selection": rejects(lambda: workbench.finalize(next(r for r in all_ids if r != selected)), "FINAL_TEST_ALREADY_SELECTED")}
        original_model = workbench.run_dir(selected) / "output/model/model.pkl"
        original_bytes = original_model.read_bytes()
        try:
            original_model.write_bytes(original_bytes + b"injected-model-mutation")
            faults["mutated_model_export"] = rejects(lambda: workbench.export(selected), "INTEGRITY_MISMATCH")
        finally:
            original_model.write_bytes(original_bytes)
        reopened = Workbench(workbench.root, dsh_context={"session_id": session_id, "call_id": "restart"})
        recovered = reopened.resume(limit=100)
        recovered_ids = {r["ref"] for r in recovered["resources"]}
        faults["backend_reopen_session_navigation"] = {"passed": set(all_ids).issubset(recovered_ids), "candidate_runs_found": len(set(all_ids) & recovered_ids)}
        label, first_spec = next(iter(specs.items()))
        repeated = reopened.create(first_spec, request_key=label)
        faults["lost_response_stable_key"] = {"passed": repeated["run_id"] == parents[0], "unique_runs": len({parents[0], repeated["run_id"]})}
        faults["same_final_selection_replay"] = {"passed": reopened.finalize(selected)["final_test"]["metrics"] == final["metrics"]}

        # Native raw operations and explicit-check positive controls, no extra final labels read.
        native_repeated = baseline.search_runs([experiment], filter_string=f"tags.request_key = '{label}'")
        random_inputs = workbench.run_dir(random_run["run_id"]) / "inputs"
        native_file = Path(url2pathname(urlsplit(baseline.get_run(native_ids[selected]).info.artifact_uri).path)) / "output/model/model.pkl"
        expected_hash, native_bytes = sha256(native_file), native_file.read_bytes()
        try:
            native_file.write_bytes(native_bytes + b"injected-model-mutation")
            native_status_unchanged = baseline.get_run(native_ids[selected]).info.status == "FINISHED"
            explicit_hash_rejects = sha256(native_file) != expected_hash
        finally:
            native_file.write_bytes(native_bytes)
        dataset_report["direct_explicit_controls"] = {
            "key_lookup_recovers_same_run": len(native_repeated) == 1 and native_repeated[0].info.run_id == native_ids[parents[0]],
            "split_hash_rejects_comparison": sha256(random_inputs / "split.json") != dataset_report["split_sha256"],
            "hash_check_detects_mutation": explicit_hash_rejects,
            "selection_tag_rejects_another_model": baseline.get_experiment(experiment).tags["final_selected_run"] != native_ids[next(r for r in all_ids if r != selected)],
            "raw_get_run_does_not_check_external_file_mutation": native_status_unchanged,
        }

        delivery = workbench.export(selected, include_data=True)
        unpacked = folder / "standalone-plugin"
        with zipfile.ZipFile(delivery["path"]) as archive:
            archive.extractall(unpacked)
        reproduced = copied_reproduction(reproduce_python, unpacked, ["reproduce.py"])
        reproduction = json.loads(reproduced.stdout.splitlines()[-1])
        faults["independent_standalone_retraining"] = {"passed": reproduction["metrics_match"], "result": reproduction}
        direct_copy = folder / "standalone-direct"
        shutil.copytree(inputs, direct_copy / "inputs")
        copied_reproduction(reproduce_python, direct_copy,
                            ["inputs/engine.py", "--inputs", "inputs", "--output", "retrained", "--final-output", "retrained-final"])
        direct_validation_match = read_json(direct_copy / "retrained/result.json")["metrics"] == next(r["metrics"] for r in dataset_report["pairs"] if r["run_id"] == selected)
        direct_final_match = read_json(direct_copy / "retrained-final/result.json")["metrics"] == final["metrics"]
        dataset_report["direct_independent_retraining"] = {"validation_match": direct_validation_match, "final_match": direct_final_match}
        assert direct_validation_match and direct_final_match
        dataset_report["workflow_checks"] = faults
        report["datasets"].append(dataset_report)
        save(args.report, report)
        print(json.dumps({"completed_dataset": name, "workflow_checks": {k: v["passed"] for k, v in faults.items()}}), flush=True)
        assert all(v["passed"] for v in faults.values())


if __name__ == "__main__":
    main()
