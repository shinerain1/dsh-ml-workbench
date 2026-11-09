import concurrent.futures
import time
import zipfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

import numpy as np
import optuna
import pandas as pd
import pytest
from conftest import wait_log, wait_run
from dsh_ml_workbench.datasets import inspect_csv, make_split
from dsh_ml_workbench.service import Workbench, owned_process
from dsh_ml_workbench.spec import ExperimentSpec, Preprocessing, SplitSpec, WorkbenchError
from optuna.trial import TrialState


def finish_search(workbench, name):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        result = workbench.search_step(name)
        if result["state"] == "complete":
            return result
        time.sleep(0.15)
    workbench.search_cancel(name)
    raise TimeoutError(name)


def test_group_and_time_boundaries_with_ties():
    frame = pd.DataFrame({"customer": np.repeat(np.arange(10), 4), "x": np.arange(40),
                          "when": np.repeat(pd.date_range("2025-01-01", periods=10).astype(str), 4),
                          "target": [0, 1] * 20})
    base = dict(dataset_path="unused.csv", task="classification", target="target")
    groups, _ = make_split(frame, ExperimentSpec(**base, split=SplitSpec(strategy="group", column="customer")))
    assert set(frame.customer.iloc[groups["train"]]).isdisjoint(frame.customer.iloc[groups["test"]])
    times, _ = make_split(frame, ExperimentSpec(**base, split=SplitSpec(strategy="time", column="when")))
    assert max(frame.when.iloc[times["train"]]) < min(frame.when.iloc[times["test"]])
    assert set(times["train"]) | set(times["test"]) == set(range(len(frame)))
    frame.loc[times["test"], "target"] = 2
    with pytest.raises(WorkbenchError, match="every target class"):
        make_split(frame, ExperimentSpec(**base, split=SplitSpec(strategy="time", column="when")))


@pytest.mark.parametrize("strategy,column", [("group", "missing"), ("time", "x"), ("time", "constant")])
def test_invalid_split_is_rejected(strategy, column):
    frame = pd.DataFrame({"x": range(20), "constant": ["2025-01-01"] * 20, "y": [0, 1] * 10})
    with pytest.raises(WorkbenchError):
        make_split(frame, ExperimentSpec(dataset_path="unused.csv", target="y", task="classification",
                                        split=SplitSpec(strategy=strategy, column=column)))


def test_real_group_fit_prediction_diff_search_and_report(tmp_path, classification_csv):
    frame = pd.read_csv(classification_csv)
    frame["customer_id"] = np.arange(len(frame)) // 4
    frame.to_csv(classification_csv, index=False)
    diagnostic = inspect_csv(str(classification_csv), "target")
    assert diagnostic["rows"] == len(frame) and "columns" in diagnostic and "data" not in diagnostic
    workbench = Workbench(tmp_path / "records")
    base = workbench.create(ExperimentSpec(dataset_path=str(classification_csv), target="target",
        name="<script>alert(1)</script>", task="classification", model="logistic_regression",
        split=SplitSpec(strategy="group", column="customer_id"), preprocessing=Preprocessing()))
    workbench.start(base["run_id"])
    result = wait_run(workbench, base["run_id"])
    assert result["state"] == "succeeded", result
    assert "customer_id" not in result["features"]
    assert {"roc_auc", "average_precision", "log_loss"} <= result["metrics"].keys()
    assert "baseline_metrics" in result and sum(map(sum, result["confusion_matrix"])) == result["test_rows"]
    unseen = tmp_path / "unseen.csv"
    frame.drop(columns=["target", "customer_id"]).iloc[:12].to_csv(unseen, index=False)
    prediction = workbench.predict(base["run_id"], str(unseen))
    assert prediction["rows"] == 12
    predicted = pd.read_csv(prediction["path"])
    assert len(predicted) == 12 and "probability_1" in predicted
    broken = frame.drop(columns=[result["features"][0]])
    broken.to_csv(unseen, index=False)
    with pytest.raises(WorkbenchError, match="Missing features"):
        workbench.predict(base["run_id"], str(unseen))
    config = dict(study_name="group-search", run_id=base["run_id"], n_trials=2,
                  search_space={"C": {"type": "float", "low": 0.01, "high": 1.0, "log": True}},
                  metric="roc_auc", reuse_preprocessing=True)
    assert workbench.search_create(**config)["budget"] == 2
    assert workbench.search_create(**config)["trials"] == []
    with pytest.raises(WorkbenchError, match="different immutable"):
        workbench.search_create(**{**config, "n_trials": 3})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        started = list(pool.map(lambda _: workbench.search_step("group-search"), range(2)))
    assert started[0]["trials"][0]["run_id"] == started[1]["trials"][0]["run_id"]
    # A fresh backend instance resumes the same persisted Optuna study.
    resumed = Workbench(workbench.root)
    final = finish_search(resumed, "group-search")
    assert len(final["trials"]) == 2 and all(t["state"] == "COMPLETE" for t in final["trials"])
    best = final["best"]["run_id"]
    diff = resumed.diff(base["run_id"], best)
    assert diff["same_evaluation"] and "params" in diff["config_changes"]
    ranking = resumed.compare([t["run_id"] for t in final["trials"]], "roc_auc")
    assert ranking["ranking"][0]["run_id"] == best
    assert resumed.list(limit=1, state="succeeded")["next_offset"] == 1
    assert len(resumed.list(parent_run_id=base["run_id"])["runs"]) == 2
    delivery = resumed.export(base["run_id"])
    with zipfile.ZipFile(delivery["path"]) as archive:
        report = archive.read("REPORT.html").decode()
        assert "&lt;script&gt;" in report and "<script>" not in report
        assert "Confusion matrix" in report and "Dummy baseline" in report


def test_optuna_crash_recovery_and_cancel(tmp_path, regression_csv, monkeypatch):
    workbench = Workbench(tmp_path / "records")
    base = workbench.create(ExperimentSpec(dataset_path=str(regression_csv), target="target",
                                           task="regression", model="ridge"))
    config = dict(study_name="recovery", run_id=base["run_id"], n_trials=2,
                  search_space={"alpha": {"type": "float", "low": 0.1, "high": 1.0}})
    created = workbench.search_create(**config)
    actual_start = workbench.start
    def crash_after_start(run_id):
        actual_start(run_id)
        raise RuntimeError("simulated lost control response")
    monkeypatch.setattr(workbench, "start", crash_after_start)
    with pytest.raises(RuntimeError, match="lost control"):
        workbench.search_step("recovery")
    current = workbench.search_get("recovery")
    run_id = current["trials"][0]["run_id"]
    monkeypatch.setattr(workbench, "start", actual_start)
    after = Workbench(workbench.root).search_step("recovery")
    assert after["trials"][0]["run_id"] == run_id
    wait_run(workbench, run_id)
    stopped = workbench.search_cancel("recovery")
    assert stopped["state"] == "cancelled"
    assert len(workbench.search_step("recovery")["trials"]) == 1
    # A crash immediately after ask cannot invent a successful trial or an experiment.
    workbench.search_create(**{**config, "study_name": "incomplete"})
    study = optuna.load_study(study_name="mlwb::incomplete", storage=created["storage_uri"])
    study.ask()
    recovered = workbench.search_step("incomplete")
    assert recovered["trials"][0]["state"] == "FAIL" and recovered["trials"][0]["run_id"] is None
    assert study.get_trials()[0].state == TrialState.FAIL


def test_mlflow_failure_cannot_commit_success(tmp_path, regression_csv, monkeypatch):
    workbench = Workbench(tmp_path / "records")
    record = workbench.create(ExperimentSpec(dataset_path=str(regression_csv), target="target", task="regression", model="ridge"))
    def failure(*args, **kwargs):
        raise RuntimeError("database unavailable")
    monkeypatch.setattr(workbench.client, "set_terminated", failure)
    with pytest.raises(RuntimeError):
        workbench._set_state(record, "succeeded")
    assert workbench.store.get(record["run_id"])["state"] == "created"


def test_study_timeout_and_cancel_stop_active_worker(tmp_path, regression_csv):
    factory = tmp_path / "slow.py"
    factory.write_text("import time\nfrom sklearn.preprocessing import StandardScaler\n"
                       "def make_preprocessor(X_train, config):\n"
                       "    print('SEARCH_READY', flush=True)\n    time.sleep(60)\n    return StandardScaler()\n", encoding="utf-8")
    workbench = Workbench(tmp_path / "records", allow_custom_code=True)
    base = workbench.create(ExperimentSpec(dataset_path=str(regression_csv), target="target",
        task="regression", model="ridge", preprocessor_path=str(factory)))
    for name in ("timed", "cancelled"):
        workbench.search_create(study_name=name, run_id=base["run_id"], n_trials=1,
            search_space={"alpha": {"type": "float", "low": 0.1, "high": 1.0}})
        started = workbench.search_step(name)
        run_id = started["trials"][0]["run_id"]
        wait_log(workbench, run_id, "SEARCH_READY")
        if name == "timed":
            with workbench.store.lock:
                record = workbench.store.get(run_id)
                record["started_at"] -= 700
                workbench.store.save(record)
            stopped = workbench.search_get(name)
            assert stopped["state"] == "complete" and stopped["trials"][0]["failure"]
        else:
            stopped = workbench.search_cancel(name)
            assert stopped["state"] == "cancelled"
        record = workbench.get(run_id)
        assert record["state"] == "cancelled" and owned_process(record) is None
        assert stopped["trials"][0]["state"] == "FAIL"


def test_optuna_recovers_committed_branch_before_run_link(tmp_path, regression_csv, monkeypatch):
    workbench = Workbench(tmp_path / "records")
    base = workbench.create(ExperimentSpec(dataset_path=str(regression_csv), target="target",
                                           task="regression", model="ridge"))
    workbench.search_create(study_name="lost-link", run_id=base["run_id"], n_trials=1,
        search_space={"alpha": {"type": "float", "low": 0.1, "high": 1.0}})
    branch = workbench.branch
    created = []
    def fail_after_commit(**kwargs):
        created.append(branch(**kwargs)["run_id"])
        raise RuntimeError("lost branch response")
    monkeypatch.setattr(workbench, "branch", fail_after_commit)
    with pytest.raises(RuntimeError, match="lost branch"):
        workbench.search_step("lost-link")
    resumed = Workbench(workbench.root)
    result = resumed.search_step("lost-link")
    assert result["trials"][0]["run_id"] == created[0]
    assert len(resumed.list(parent_run_id=base["run_id"])["runs"]) == 1
    assert resumed.search_cancel("lost-link")["state"] == "cancelled"


def test_mlflow_artifact_copy_must_match_exported_model(tmp_path, regression_csv):
    workbench = Workbench(tmp_path / "records")
    record = workbench.create(ExperimentSpec(dataset_path=str(regression_csv), target="target",
                                           task="regression", model="ridge"))
    workbench.start(record["run_id"])
    assert wait_run(workbench, record["run_id"])["state"] == "succeeded"
    uri = workbench.client.get_run(record["run_id"]).info.artifact_uri
    artifact_root = Path(url2pathname(urlsplit(uri).path))
    remote_output = artifact_root / "output"
    model_file = remote_output / "model" / "model.pkl"
    original = model_file.read_bytes()
    model_file.write_bytes(original + b"changed")
    with pytest.raises(WorkbenchError, match="Snapshot files changed"):
        workbench.export(record["run_id"])
    model_file.write_bytes(original)
    remote_inputs = artifact_root / "inputs"
    (remote_inputs / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(WorkbenchError, match="Snapshot files changed"):
        workbench.get(record["run_id"])
