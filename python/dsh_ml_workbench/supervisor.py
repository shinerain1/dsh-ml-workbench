"""Own a training process and enforce its persisted wall-clock deadline without RPC polling."""

import os
import subprocess
import sys
import time

import psutil

from .service import Workbench


def stop_tree(process):
    """Terminate only this live child and its descendants; confirm exit before reporting failure."""
    try:
        targets = [*process.children(recursive=True), process]
    except psutil.NoSuchProcess:
        return
    for target in targets:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(targets, timeout=5)
    if alive:
        raise RuntimeError("Training process range did not exit after timeout")


def supervise(root, run_id, token):
    workbench = Workbench(root, allow_custom_code=True)
    with workbench.store.lock:
        record = workbench.store.get(run_id)
        if record.get("launch_token") != token or record["state"] != "starting":
            return 0
        # Windows venv launchers may hand off to another PID. The atomic state/token
        # claim, not the launcher's PID, prevents a second supervisor from owning it.
        record.update(pid=os.getpid(), process_created_at=psutil.Process().create_time())
        workbench._set_state(record, "running")
    process = None
    try:
        remaining = record["deadline_at"] - time.time()
        if remaining <= 0:
            raise TimeoutError("TRAINING_TIMEOUT: deadline elapsed during startup")
        argv = [sys.executable, "-m", "dsh_ml_workbench", "worker", "--root", str(workbench.root),
                "--run-id", run_id, "--token", token, "--parent-pid", str(os.getpid()),
                "--parent-created", str(record["process_created_at"])]
        with (workbench.run_dir(run_id) / "worker.log").open("ab", buffering=0) as log:
            process = psutil.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   cwd=workbench.root, close_fds=True,
                                   **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        try:
            process.wait(timeout=max(0.01, record["deadline_at"] - time.time()))
        except psutil.TimeoutExpired:
            stop_tree(process)
            raise TimeoutError("TRAINING_TIMEOUT: persisted training deadline exceeded; training process terminated") from None
        with workbench.store.lock:
            current = workbench.store.get(run_id)
            if current["state"] == "running":
                workbench._set_state(current, "failed", "Training process exited without committing output")
        return 0
    except Exception as error:
        if process is not None:
            stop_tree(process)
        with workbench.store.lock:
            current = workbench.store.get(run_id)
            if current["state"] in {"starting", "running"}:
                workbench._set_state(current, "failed", f"{type(error).__name__}: {error}"[:1000])
        return 1
