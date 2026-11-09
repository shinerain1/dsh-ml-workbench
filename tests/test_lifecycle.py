import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import psutil
from conftest import wait_log, wait_run
from dsh_ml_workbench.service import Workbench, owned_process
from dsh_ml_workbench.spec import ExperimentSpec


def slow_spec(tmp_path, data):
    source = tmp_path / "slow_estimator.py"
    source.write_text(
        "import time\nfrom sklearn.linear_model import LogisticRegression\n"
        "def make_estimator(task, params, seed):\n"
        "    print('FACTORY_ENTERED', flush=True)\n"
        "    time.sleep(120)\n"
        "    return LogisticRegression(random_state=seed)\n", encoding="utf-8")
    return ExperimentSpec(dataset_path=str(data), target="target", task="classification",
                          model="custom", estimator_path=str(source))


def test_concurrent_start_has_one_worker_and_cancel_waits_for_exit(tmp_path, classification_csv):
    workbench = Workbench(tmp_path / "workbench", allow_custom_code=True)
    run = workbench.create(slow_spec(tmp_path, classification_csv))
    try:
        def start():
            return Workbench(workbench.root, allow_custom_code=True).start(run["run_id"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: start(), range(2)))
        assert results[0]["pid"] == results[1]["pid"]
        active = wait_log(workbench, run["run_id"], "FACTORY_ENTERED")
        assert owned_process(active) is not None
        cancelled = workbench.cancel(run["run_id"])
        assert cancelled["state"] == "cancelled"
        assert owned_process(cancelled) is None
        assert workbench.start(run["run_id"])["state"] == "cancelled"
        assert [event["state"] for event in workbench.get(run["run_id"])["events"]].count("running") == 1
    finally:
        workbench.cancel(run["run_id"])


def test_worker_crash_is_detected_and_restart_uses_new_branch(tmp_path, classification_csv):
    workbench = Workbench(tmp_path / "workbench", allow_custom_code=True)
    run = workbench.create(slow_spec(tmp_path, classification_csv))
    workbench.start(run["run_id"])
    try:
        active = wait_log(workbench, run["run_id"], "FACTORY_ENTERED")
        process = owned_process(active)
        process.kill()
        process.wait(timeout=10)
        recovered = Workbench(workbench.root, allow_custom_code=True).get(run["run_id"])
        assert recovered["state"] == "interrupted"
        branch = workbench.branch(run["run_id"], {"model": "logistic_regression", "estimator_path": None},
                                  reuse_preprocessing=False)
        assert branch["run_id"] != run["run_id"]
        workbench.start(branch["run_id"])
        assert wait_run(workbench, branch["run_id"])["state"] == "succeeded"
        assert workbench.get(run["run_id"])["state"] == "interrupted"
    finally:
        workbench.cancel(run["run_id"])


def test_rpc_exit_does_not_kill_training_and_protocol_is_versioned(tmp_path, classification_csv):
    workbench = Workbench(tmp_path / "workbench")
    run = workbench.create(ExperimentSpec(dataset_path=str(classification_csv), target="target", task="classification", model="logistic_regression"))
    invocation = subprocess.run([sys.executable, "-m", "dsh_ml_workbench", "rpc", "--root", str(workbench.root)],
        input=json.dumps({"protocol_version": 1, "operation": "start", "args": {"run_id": run["run_id"]}}),
        capture_output=True, text=True, timeout=60)
    assert invocation.returncode == 0, invocation.stderr
    assert json.loads(invocation.stdout)["ok"] is True
    assert wait_run(Workbench(workbench.root), run["run_id"])["state"] == "succeeded"
    invalid = subprocess.run([sys.executable, "-m", "dsh_ml_workbench", "rpc", "--root", str(workbench.root)],
        input=json.dumps({"protocol_version": 9, "operation": "start", "args": {"run_id": run["run_id"]}}),
        capture_output=True, text=True, timeout=60)
    assert json.loads(invalid.stdout)["ok"] is False


def test_pid_reuse_never_targets_an_unrelated_process():
    current = psutil.Process()
    assert owned_process({"pid": current.pid, "process_created_at": current.create_time() - 100}) is None
    assert current.is_running()
