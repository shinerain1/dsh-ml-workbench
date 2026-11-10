import json
import subprocess
import sys
import zipfile

import mlflow.sklearn
import pandas as pd
import pytest
from conftest import wait_run
from dsh_ml_workbench.files import read_json, sha256
from dsh_ml_workbench.service import Workbench
from dsh_ml_workbench.spec import ExperimentSpec, WorkbenchError


def create(workbench, data, **kwargs):
    return workbench.create(ExperimentSpec(dataset_path=str(data), target="target", task="classification", **kwargs))


def test_classification_branch_model_consistency_and_standalone_delivery(tmp_path, classification_csv, monkeypatch):
    workbench = Workbench(tmp_path / "workbench")
    parent = create(workbench, classification_csv, model="random_forest", params={"n_estimators": 12})
    workbench.start(parent["run_id"])
    parent_done = wait_run(workbench, parent["run_id"])
    assert parent_done["state"] == "succeeded", parent_done
    folder = workbench.run_dir(parent["run_id"])
    parent_manifest_hash = sha256(folder / "manifest.json")
    parent_model = mlflow.sklearn.load_model(str(folder / "output" / "model"))
    frame = pd.read_csv(classification_csv)
    split = read_json(folder / "inputs" / "split.json")
    predictions = pd.read_csv(folder / "output" / "predictions.csv")
    assert parent_model.predict(frame.drop(columns="target").iloc[split["test"]]).tolist() == predictions.prediction.tolist()
    # The original input is no longer needed after snapshotting.
    classification_csv.unlink()
    child = workbench.branch(parent["run_id"], {"name": "faster", "model": "logistic_regression", "params": {}})
    child_dir = workbench.run_dir(child["run_id"])
    assert sha256(child_dir / "inputs" / "reused_preprocessor.pkl") == sha256(folder / "output" / "preprocessor.pkl")
    workbench.start(child["run_id"])
    child_done = wait_run(Workbench(workbench.root), child["run_id"])
    assert child_done["state"] == "succeeded", child_done
    assert child_done["reused_preprocessing"] is True
    assert sha256(folder / "manifest.json") == parent_manifest_hash
    comparison = workbench.compare([parent["run_id"], child["run_id"]])
    assert comparison["metric"] == "accuracy" and len(comparison["ranking"]) == 2
    tracked = workbench.client.get_run(child["run_id"])
    assert tracked.data.tags["mlflow.parentRunId"] == parent["run_id"]
    assert tracked.data.metrics["accuracy"] == child_done["metrics"]["accuracy"]
    artifact_names = {item.path for item in workbench.client.list_artifacts(child["run_id"], "output")}
    assert "output/model" in artifact_names
    assert "final_test" not in child_done
    with monkeypatch.context() as patch:
        def fail_final_artifacts(*args, **kwargs):
            raise RuntimeError("simulated final artifact commit failure")
        patch.setattr(workbench.client, "log_artifacts", fail_final_artifacts)
        with pytest.raises(RuntimeError, match="simulated final artifact"):
            workbench.finalize(child["run_id"])
    pending_predictions = child_dir / "final-test" / "predictions.csv"
    original_predictions = pending_predictions.read_bytes()
    pending_predictions.write_bytes(original_predictions + b"corrupt\n")
    with pytest.raises(WorkbenchError, match="Uncommitted final-test output differs"):
        workbench.finalize(child["run_id"])
    pending_predictions.write_bytes(original_predictions)
    final = workbench.finalize(child["run_id"])
    assert final["final_test"]["rows"] == len(split["final_test"])
    assert workbench.finalize(child["run_id"]) == final
    with pytest.raises(WorkbenchError) as caught:
        workbench.finalize(parent["run_id"])
    assert caught.value.code == "FINAL_TEST_ALREADY_SELECTED"
    delivery = workbench.export(child["run_id"], include_data=True)
    unpacked = tmp_path / "delivery"
    with zipfile.ZipFile(delivery["path"]) as archive:
        archive.extractall(unpacked)
    verified = subprocess.run([sys.executable, "reproduce.py", "--verify-only"], cwd=unpacked,
                              capture_output=True, text=True, timeout=30)
    assert verified.returncode == 0, verified.stderr
    repeated = subprocess.run([sys.executable, "reproduce.py"], cwd=unpacked,
                             capture_output=True, text=True, timeout=90)
    assert repeated.returncode == 0, repeated.stdout + repeated.stderr
    assert json.loads(repeated.stdout.splitlines()[-1])["metrics_match"]
    assert json.loads(repeated.stdout.splitlines()[-1])["final_test_metrics_match"] is True
    private_delivery = workbench.export(child["run_id"])
    with zipfile.ZipFile(private_delivery["path"]) as archive:
        assert "inputs/dataset.csv" not in archive.namelist()
    final_file = child_dir / "final-test" / "predictions.csv"
    final_file.write_text("changed", encoding="utf-8")
    with pytest.raises(WorkbenchError, match="Final-test artifacts changed"):
        workbench.export(child["run_id"])


def test_regression_and_incompatible_evaluation_are_not_ranked(tmp_path, regression_csv):
    workbench = Workbench(tmp_path / "workbench")
    first = workbench.create(ExperimentSpec(dataset_path=str(regression_csv), target="target", task="regression", model="ridge"))
    workbench.start(first["run_id"])
    assert wait_run(workbench, first["run_id"])["state"] == "succeeded"
    second = workbench.branch(first["run_id"], {"model": "random_forest", "params": {"n_estimators": 10}})
    workbench.start(second["run_id"])
    assert wait_run(workbench, second["run_id"])["state"] == "succeeded"
    comparison = workbench.compare([first["run_id"], second["run_id"]], "rmse")
    assert comparison["direction"] == "min"
    assert comparison["ranking"][0]["metrics"]["rmse"] <= comparison["ranking"][1]["metrics"]["rmse"]
    with pytest.raises(WorkbenchError, match="identical data"):
        workbench.branch(first["run_id"], {"seed": 123})
    incompatible = workbench.branch(first["run_id"], {"seed": 123}, reuse_preprocessing=False)
    workbench.start(incompatible["run_id"])
    assert wait_run(workbench, incompatible["run_id"])["state"] == "succeeded"
    with pytest.raises(WorkbenchError) as caught:
        workbench.compare([first["run_id"], incompatible["run_id"]])
    assert caught.value.code == "INCOMPARABLE_EXPERIMENTS"


def test_idempotency_and_input_tampering(tmp_path, classification_csv):
    workbench = Workbench(tmp_path / "workbench")
    spec = ExperimentSpec(dataset_path=str(classification_csv), target="target", task="classification")
    first = workbench.create(spec, "request-1")
    assert workbench.create(spec, "request-1")["run_id"] == first["run_id"]
    with pytest.raises(WorkbenchError) as caught:
        workbench.create(spec.model_copy(update={"seed": 43}), "request-1")
    assert caught.value.code == "IDEMPOTENCY_CONFLICT"
    engine = workbench.run_dir(first["run_id"]) / "inputs" / "engine.py"
    engine.write_text(engine.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    with pytest.raises(WorkbenchError) as caught:
        workbench.start(first["run_id"])
    assert caught.value.code == "INTEGRITY_MISMATCH"
    assert workbench.get(first["run_id"])["state"] == "created"


def test_tracking_mismatch_blocks_delivery(tmp_path, classification_csv):
    workbench = Workbench(tmp_path / "workbench")
    run = create(workbench, classification_csv, model="logistic_regression")
    workbench.start(run["run_id"])
    assert wait_run(workbench, run["run_id"])["state"] == "succeeded"
    workbench.client.log_metric(run["run_id"], "accuracy", -1.0)
    with pytest.raises(WorkbenchError) as caught:
        workbench.export(run["run_id"])
    assert caught.value.code == "TRACKING_MISMATCH"


def test_invalid_dataset_and_disabled_custom_code(tmp_path, classification_csv):
    workbench = Workbench(tmp_path / "workbench")
    code = tmp_path / "custom.py"
    code.write_text("raise RuntimeError('must not execute during creation')\n", encoding="utf-8")
    with pytest.raises(WorkbenchError) as caught:
        create(workbench, classification_csv, preprocessor_path=str(code))
    assert caught.value.code == "CUSTOM_CODE_DISABLED"
    duplicate = tmp_path / "duplicates.csv"
    duplicate.write_text("x,x,target\n1,2,0\n" * 10, encoding="utf-8")
    with pytest.raises(WorkbenchError, match="unique"):
        create(workbench, duplicate)
