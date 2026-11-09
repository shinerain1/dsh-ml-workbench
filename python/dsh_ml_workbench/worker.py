"""A detached local worker continues when the DSH session or RPC caller closes."""

import importlib.util
import os
import sys
import traceback
from pathlib import Path

import psutil

from .files import atomic_json, digest, tree_hashes
from .service import Workbench


def run_worker(root: Path, run_id: str, token: str) -> int:
    workbench = Workbench(root, allow_custom_code=True)
    store = workbench.store
    with store.lock:
        record = store.get(run_id)
        if record.get("launch_token") != token or record["state"] != "starting":
            return 0
        record.update(pid=os.getpid(), process_created_at=psutil.Process().create_time())
        workbench._set_state(record, "running")
    try:
        workbench._validate_inputs(record)
        folder = workbench.run_dir(run_id)
        sys.dont_write_bytecode = True
        module_spec = importlib.util.spec_from_file_location("mlwb_recipe", folder / "inputs" / "engine.py")
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError("Cannot load the snapshotted training recipe")
        recipe = importlib.util.module_from_spec(module_spec)
        sys.modules["mlwb_recipe"] = recipe
        module_spec.loader.exec_module(recipe)
        result = recipe.train(folder / "inputs", folder / "output.pending")
        with store.lock:
            record = store.get(run_id)
            if record["state"] != "running" or record["launch_token"] != token:
                return 0
            workbench._validate_inputs(record)
            for key, value in result["metrics"].items():
                workbench.client.log_metric(run_id, key, value)
            workbench.client.log_metric(run_id, "train_seconds", result["train_seconds"])
            workbench.client.log_metric(run_id, "predict_seconds", result["predict_seconds"])
            workbench.client.log_artifacts(run_id, str(folder / "output.pending"), "output")
            (folder / "output.pending").rename(folder / "output")
            hashes = tree_hashes(folder / "output")
            atomic_json(folder / "output-checksums.json", hashes)
            record["output_hash"] = digest(hashes)
            workbench._set_state(record, "succeeded")
        return 0
    except Exception as error:
        traceback.print_exc()
        with store.lock:
            record = store.get(run_id)
            if record["state"] in {"running", "starting"}:
                workbench._set_state(record, "failed", f"{type(error).__name__}: {error}"[:1000])
        return 1
