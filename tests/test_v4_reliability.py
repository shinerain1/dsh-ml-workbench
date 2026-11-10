
import psutil
import pytest
from conftest import wait_log
from dsh_ml_workbench.__main__ import dispatch
from dsh_ml_workbench.datasets import make_split, read_csv
from dsh_ml_workbench.service import Workbench, owned_process
from dsh_ml_workbench.spec import ExperimentSpec, RpcRequest, WorkbenchError
from test_lifecycle import slow_spec


def test_three_part_split_is_disjoint_reproducible_and_legacy_compatible(classification_csv):
    frame = read_csv(classification_csv)
    spec = ExperimentSpec(dataset_path=str(classification_csv), target="target", task="classification")
    split, _ = make_split(frame, spec)
    assert split == make_split(frame, spec)[0]
    assert sorted(sum(split.values(), [])) == list(range(len(frame)))
    assert [len(split[key]) for key in ("train", "test", "final_test")] == [96, 32, 32]
    legacy, _ = make_split(frame, spec.model_copy(update={"final_test_fraction": 0.0}))
    assert set(legacy) == {"train", "test"} and len(legacy["train"]) == 128


def test_training_deadline_fires_without_get_or_search_polling(tmp_path, classification_csv):
    workbench = Workbench(tmp_path / "records", allow_custom_code=True)
    spec = slow_spec(tmp_path, classification_csv).model_copy(update={"timeout_seconds": 20})
    run = workbench.create(spec)
    started = workbench.start(run["run_id"])
    process = owned_process(started)
    try:
        # This observes process exit only. No workbench reconciliation or tuning poll runs.
        process.wait(timeout=40)
        record = workbench.store.get(run["run_id"])
        assert record["state"] == "failed" and "TRAINING_TIMEOUT" in record["error"]
        assert "FACTORY_ENTERED" in (workbench.run_dir(run["run_id"]) / "worker.log").read_text(encoding="utf-8")
        assert owned_process({"pid": record["training_pid"], "process_created_at": record["training_process_created_at"]}) is None
        assert workbench.client.get_run(run["run_id"]).info.status == "FAILED"
    finally:
        workbench.cancel(run["run_id"])


def test_supervisor_death_does_not_leave_training_orphan(tmp_path, classification_csv):
    workbench = Workbench(tmp_path / "records", allow_custom_code=True)
    run = workbench.create(slow_spec(tmp_path, classification_csv))
    workbench.start(run["run_id"])
    record = wait_log(workbench, run["run_id"], "FACTORY_ENTERED")
    supervisor = owned_process(record)
    child = psutil.Process(record["training_pid"])
    supervisor.kill()
    supervisor.wait(10)
    child.wait(10)
    assert workbench.get(run["run_id"])["state"] == "interrupted"


def test_version_mismatch_and_external_launch_retry(tmp_path, regression_csv):
    workbench = Workbench(tmp_path / "records", external_workers=True)
    with pytest.raises(WorkbenchError, match="Plugin expects"):
        dispatch(workbench, RpcRequest(operation="health", expected_backend_version="0.0.0"))
    run = dispatch(workbench, RpcRequest(operation="run", args={"dataset_path": str(regression_csv),
        "target": "target", "task": "regression", "model": "ridge", "request_key": "combined-run"}))
    repeated = workbench.start(run["run_id"])
    assert run["launch_token"] == repeated["launch_token"]
    assert workbench.pending_launches[0] == workbench.pending_launches[1]
    assert run["pid"] is None and workbench.wait(run["run_id"], 0)["state"] == "starting"
    assert workbench.cancel(run["run_id"])["state"] == "cancelled"
