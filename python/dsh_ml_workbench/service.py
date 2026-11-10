"""Agent workflow around MLflow, with immutable inputs and explicit worker ownership."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

import numpy as np
import pandas as pd
import psutil
from mlflow import MlflowClient
from sklearn.utils.multiclass import type_of_target

from . import __version__
from .datasets import MAX_DATASET_BYTES, inspect_csv, make_split, read_csv
from .files import (
    atomic_json,
    canonical,
    digest,
    environment,
    read_json,
    requirements,
    sha256,
    tree_hashes,
    verify_tree,
)
from .spec import DshContext, ExperimentSpec, ResumeRequest, WorkbenchError
from .store import TERMINAL, Store

STARTUP_GRACE_SECONDS = 30
METRICS = {"accuracy": "max", "balanced_accuracy": "max", "f1_macro": "max",
           "roc_auc": "max", "average_precision": "max", "log_loss": "min",
           "rmse": "min", "mae": "min", "r2": "max"}


def owned_process(record: dict):
    """Match PID and creation time before observing or terminating a worker."""
    if record.get("pid") is None:
        return None
    try:
        process = psutil.Process(record["pid"])
        if abs(process.create_time() - record["process_created_at"]) > 0.01:
            return None
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        return process
    except psutil.NoSuchProcess:
        return None
    except psutil.AccessDenied as error:
        raise WorkbenchError("PROCESS_STATE_UNKNOWN", "Cannot inspect worker ownership; no process was killed") from error


class Workbench:
    def __init__(self, root: Path, *, allow_custom_code: bool = False, dsh_context: dict | None = None,
                 external_workers: bool = False):
        self.external_workers = external_workers
        self.pending_launches = []
        self.dsh_context = DshContext.model_validate(dsh_context).model_dump() if dsh_context is not None else None
        self.store = Store(root)
        self.root = self.store.root
        self.allow_custom_code = allow_custom_code
        self.tracking_uri = "sqlite:///" + (self.root / "mlflow.sqlite3").as_posix()
        # Serialise initial MLflow schema migrations across independent control processes.
        with self.store.lock:
            self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def run_dir(self, run_id: str) -> Path:
        self.store.get(run_id)
        return self.root / "runs" / run_id

    def _set_state(self, record: dict, state: str, error: str | None = None) -> dict:
        record["state"] = state
        if error is not None:
            record["error"] = error
        self.client.set_tag(record["run_id"], "mlwb.state", state)
        if state in TERMINAL:
            status = "FINISHED" if state == "succeeded" else "KILLED" if state == "cancelled" else "FAILED"
            self.client.set_terminated(record["run_id"], status)
        # Never persist a successful lifecycle claim before MLflow has committed it.
        self.store.save(record)
        return record

    def _validate_inputs(self, record: dict) -> dict:
        run_dir = self.run_dir(record["run_id"])
        manifest = read_json(run_dir / "manifest.json")
        if digest(manifest) != record["manifest_hash"]:
            raise WorkbenchError("INTEGRITY_MISMATCH", "Experiment manifest changed")
        verify_tree(run_dir / "inputs", manifest["files"])
        return manifest

    def _validate_outputs(self, record: dict) -> dict:
        self._validate_inputs(record)
        if record["state"] != "succeeded":
            raise WorkbenchError("RUN_NOT_SUCCEEDED", "Only a succeeded experiment can be compared or exported")
        folder = self.run_dir(record["run_id"])
        checksums = read_json(folder / "output-checksums.json")
        if digest(checksums) != record["output_hash"]:
            raise WorkbenchError("INTEGRITY_MISMATCH", "Output checksum record changed")
        verify_tree(folder / "output", checksums)
        result = read_json(folder / "output" / "result.json")
        tracked = self.client.get_run(record["run_id"])
        if tracked.info.status != "FINISHED" or any(
            tracked.data.metrics.get(key) != value for key, value in result["metrics"].items()
        ):
            raise WorkbenchError("TRACKING_MISMATCH", "MLflow status or metrics disagree with the recorded model output")
        # The runs:/ model URI must serve the same files as an exported local model.
        artifact_uri = urlsplit(tracked.info.artifact_uri)
        if artifact_uri.scheme != "file" or artifact_uri.netloc not in {"", "localhost"} or artifact_uri.query or artifact_uri.fragment:
            raise WorkbenchError("TRACKING_MISMATCH", "This local backend requires its original local MLflow artifact store")
        artifact_root = Path(url2pathname(artifact_uri.path)).resolve()
        if not artifact_root.is_relative_to((self.root / "mlflow-artifacts").resolve()):
            raise WorkbenchError("TRACKING_MISMATCH", "MLflow artifact location moved outside this workbench root")
        verify_tree(artifact_root / "output", checksums)
        manifest = read_json(folder / "manifest.json")
        verify_tree(artifact_root / "inputs", manifest["files"])
        return result

    def create(self, spec: ExperimentSpec, request_key: str | None = None, *, parent: dict | None = None,
               reuse_preprocessing: bool = False, payload_hash: str | None = None) -> dict:
        payload_hash = payload_hash or digest({"operation": "create", "spec": spec.model_dump()})
        with self.store.lock:
            previous = self.store.previous_request(request_key, payload_hash)
            if previous is not None:
                self.store.link(self.dsh_context, "run", previous["run_id"])
                return previous
            has_custom = spec.preprocessor_path is not None or spec.estimator_path is not None
            if has_custom and not self.allow_custom_code:
                raise WorkbenchError("CUSTOM_CODE_DISABLED", "Enable allowCustomCode to execute reviewed Python factories")
            source = Path(spec.dataset_path).expanduser().resolve()
            if not source.is_file() or source.stat().st_size > MAX_DATASET_BYTES:
                raise WorkbenchError("INVALID_DATASET", "dataset_path must be a local CSV file no larger than 128 MiB")
            staging = self.root / "staging" / uuid.uuid4().hex
            inputs = staging / "inputs"
            inputs.mkdir(parents=True)
            run_id = None
            try:
                shutil.copyfile(source, inputs / "dataset.csv")
                if (inputs / "dataset.csv").stat().st_size > MAX_DATASET_BYTES:
                    raise WorkbenchError("INVALID_DATASET", "Dataset exceeded 128 MiB while copying")
                frame = read_csv(inputs / "dataset.csv")
                if spec.target not in frame or len(frame) < 10:
                    raise WorkbenchError("INVALID_DATASET", "Target must exist and the CSV must contain at least 10 rows")
                if frame[spec.target].isna().any():
                    raise WorkbenchError("INVALID_TARGET", "Target contains missing values; clean labels explicitly")
                missing = set(spec.preprocessing.drop_columns) - set(frame.columns)
                if missing or len(set(frame.columns) - {spec.target, spec.split.column, *spec.preprocessing.drop_columns}) == 0:
                    raise WorkbenchError("INVALID_FEATURES", "Unknown drop column or no feature columns remain")
                numeric = frame.select_dtypes(include=[np.number])
                if np.isinf(numeric.to_numpy()).any():
                    raise WorkbenchError("INVALID_DATASET", "Infinite values must be cleaned explicitly")
                labels = frame[spec.target]
                if spec.task == "classification":
                    if type_of_target(labels) not in {"binary", "multiclass"} or labels.nunique() < 2:
                        raise WorkbenchError("INVALID_TARGET", "Classification requires at least two discrete classes")
                elif not pd.api.types.is_numeric_dtype(labels):
                    raise WorkbenchError("INVALID_TARGET", "Regression requires a numeric target")
                split, split_details = make_split(frame, spec)
                atomic_json(inputs / "split.json", split)
                dataset_hash = sha256(inputs / "dataset.csv")
                portable = spec.model_dump()
                portable["dataset_path"] = "dataset.csv"
                if spec.split.column is not None:
                    portable["preprocessing"]["drop_columns"] = sorted(set(
                        [*spec.preprocessing.drop_columns, spec.split.column]))
                features = [name for name in frame if name not in {spec.target, *portable["preprocessing"]["drop_columns"]}]
                atomic_json(inputs / "dataset.json", {"sha256": dataset_hash, "rows": len(frame),
                    "columns": frame.columns.tolist(), "parser": "pandas-csv-v1", "split": split_details,
                    "feature_schema": {name: "numeric" if pd.api.types.is_numeric_dtype(frame[name]) else "categorical" for name in features}})
                for field, filename in [("preprocessor_path", "preprocessor.py"), ("estimator_path", "estimator.py")]:
                    path = getattr(spec, field)
                    if path is not None:
                        code = Path(path).expanduser().resolve()
                        if not code.is_file() or code.suffix != ".py" or code.stat().st_size > 1024 * 1024:
                            raise WorkbenchError("INVALID_CODE", "Factory must be a Python file no larger than 1 MiB")
                        shutil.copyfile(code, inputs / filename)
                        portable[field] = filename
                atomic_json(inputs / "config.json", portable)
                env = environment()
                atomic_json(inputs / "environment.json", env)
                (inputs / "requirements.txt").write_text(requirements(), encoding="utf-8")
                shutil.copyfile(Path(__file__).with_name("engine.py"), inputs / "engine.py")
                parent_contract = None
                if reuse_preprocessing and parent is not None:
                    parent_inputs = self.run_dir(parent["run_id"]) / "inputs"
                    if read_json(parent_inputs / "environment.json")["packages"] != env["packages"]:
                        raise WorkbenchError("PREPROCESSING_MISMATCH", "Fitted preprocessing requires identical package versions")
                    shutil.copyfile(parent_inputs / "engine.py", inputs / "engine.py")
                    parent_contract = self._validate_inputs(parent)["contract"]
                contract = {"dataset_sha256": dataset_hash, "split_sha256": digest(split),
                            "target": spec.target, "task": spec.task, "evaluator": "holdout-v2",
                            "metric_library": env["packages"]["scikit-learn"], "parser": "pandas-csv-v1",
                            "pandas": env["packages"]["pandas"], "numpy": env["packages"]["numpy"],
                            "recipe_sha256": sha256(inputs / "engine.py")}
                if parent_contract is not None:
                    # Keep legacy evaluator identity when reusing a legacy fitted pipeline.
                    contract = {**parent_contract, **{key: contract[key] for key in
                        ("dataset_sha256", "split_sha256", "target", "task", "metric_library", "parser")}}
                preprocessor_hash = digest({"config": portable["preprocessing"],
                    "source": sha256(inputs / "preprocessor.py") if has_custom and spec.preprocessor_path else None,
                    "recipe": sha256(inputs / "engine.py")})
                if reuse_preprocessing:
                    if parent is None:
                        raise WorkbenchError("INVALID_BRANCH", "Reusing preprocessing requires a parent")
                    self._validate_outputs(parent)
                    if parent["comparison_key"] != digest(contract) or parent["preprocessing_hash"] != preprocessor_hash:
                        raise WorkbenchError("PREPROCESSING_MISMATCH", "Fitted preprocessing requires identical data, split and preprocessing code/config")
                    shutil.copyfile(self.run_dir(parent["run_id"]) / "output" / "preprocessor.pkl",
                                    inputs / "reused_preprocessor.pkl")
                manifest = {"format": 1, "contract": contract, "files": tree_hashes(inputs),
                            "parent_run_id": parent["run_id"] if parent else None,
                            "preprocessing_hash": preprocessor_hash}
                if self.dsh_context is not None:
                    manifest["dsh_context"] = self.dsh_context
                atomic_json(staging / "manifest.json", manifest)
                experiment = self.client.get_experiment_by_name(spec.experiment_name)
                experiment_id = experiment.experiment_id if experiment else self.client.create_experiment(
                    spec.experiment_name, artifact_location=(self.root / "mlflow-artifacts").as_uri())
                tags = {"mlwb.version": __version__, "mlwb.state": "created", "mlwb.comparison_key": digest(contract),
                        "mlwb.dataset_sha256": dataset_hash, "mlwb.preprocessing_hash": preprocessor_hash}
                if self.dsh_context is not None:
                    tags.update({"mlwb.dsh." + key: value for key, value in self.dsh_context.items()})
                if parent:
                    tags["mlflow.parentRunId"] = parent["run_id"]
                run_id = self.client.create_run(experiment_id, tags=tags, run_name=spec.name).info.run_id
                destination = self.root / "runs" / run_id
                destination.parent.mkdir(exist_ok=True)
                staging.rename(destination)
                self.client.log_param(run_id, "task", spec.task)
                self.client.log_param(run_id, "model", spec.model)
                self.client.log_param(run_id, "seed", spec.seed)
                self.client.log_param(run_id, "test_fraction", spec.test_fraction)
                for key, value in spec.params.items():
                    self.client.log_param(run_id, "model." + key, canonical(value))
                self.client.log_artifacts(run_id, str(destination / "inputs"), "inputs")
                self.client.log_artifact(run_id, str(destination / "manifest.json"))
                record = {"run_id": run_id, "name": spec.name, "experiment_id": experiment_id, "state": "created",
                          "created_at": time.time(), "manifest_hash": digest(manifest), "comparison_key": digest(contract),
                          "preprocessing_hash": preprocessor_hash, "parent_run_id": parent["run_id"] if parent else None,
                          "task": spec.task, "model": spec.model, "custom_code": has_custom, "pid": None,
                          "reused_preprocessing": reuse_preprocessing}
                if self.dsh_context is not None:
                    record["dsh_context"] = self.dsh_context
                self.store.save(record, request_key=request_key, payload_hash=payload_hash)
                return record
            except Exception:
                if run_id is not None:
                    self.client.set_terminated(run_id, "FAILED")
                raise
            finally:
                # Only this call's UUID staging directory is disposable.
                if staging.exists():
                    shutil.rmtree(staging)

    def start(self, run_id: str, *, timeout_seconds: int | None = None) -> dict:
        with self.store.lock:
            record = self._get_locked(run_id)
            if record["state"] != "created":
                if record["state"] == "starting" and record.get("external_worker") and record.get("pid") is None:
                    self.pending_launches.append({"run_id": run_id, "token": record["launch_token"]})
                return record
            self._validate_inputs(record)
            if record["custom_code"] and not self.allow_custom_code:
                raise WorkbenchError("CUSTOM_CODE_DISABLED", "Enable allowCustomCode before starting this experiment")
            token = uuid.uuid4().hex
            config = read_json(self.run_dir(run_id) / "inputs" / "config.json")
            timeout = timeout_seconds if timeout_seconds is not None else config.get("timeout_seconds", 3600)
            if type(timeout) is not int or not 5 <= timeout <= 86400:
                raise WorkbenchError("INVALID_TIMEOUT", "timeout_seconds must be an integer from 5 to 86400")
            record.update(launch_token=token, started_at=time.time(), timeout_seconds=timeout,
                          deadline_at=time.time() + timeout, external_worker=self.external_workers)
            self._set_state(record, "starting")
            if self.external_workers:
                self.pending_launches.append({"run_id": run_id, "token": token})
                return record
            argv = [sys.executable, "-m", "dsh_ml_workbench", "supervise", "--root", str(self.root),
                    "--run-id", run_id, "--token", token]
            env = dict(os.environ, PYTHONUTF8="1", MLFLOW_ENABLE_TELEMETRY="false")
            options = {"start_new_session": True} if os.name != "nt" else {
                "creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            }
            try:
                with (self.run_dir(run_id) / "worker.log").open("ab", buffering=0) as log:
                    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                               cwd=self.root, env=env, close_fds=True, **options)
                record.update(pid=process.pid, process_created_at=psutil.Process(process.pid).create_time())
                self.store.save(record)
            except Exception as error:
                self._set_state(record, "failed", f"Worker launch failed: {type(error).__name__}: {error}")
                raise WorkbenchError("LAUNCH_FAILED", record["error"]) from error
            return record

    def _get_locked(self, run_id: str) -> dict:
        record = self.store.get(run_id)
        if record["state"] in {"starting", "running", "cancelling"}:
            grace = record["state"] == "starting" and time.time() - record["started_at"] < STARTUP_GRACE_SECONDS
            if not grace and owned_process(record) is None:
                state = "cancelled" if record["state"] == "cancelling" else "interrupted"
                self._set_state(record, state, "Worker exited without committing a successful result; branch to restart")
        return record

    def get(self, run_id: str) -> dict:
        with self.store.lock:
            record = self._get_locked(run_id)
            if record["state"] == "succeeded":
                record = {**record, **self._validate_outputs(record),
                          "model_uri": f"runs:/{run_id}/output/model"}
                if record.get("final_hash"):
                    record["final_test"] = self._validate_final(record)
            record["events"] = self.store.events(run_id)
            log = self.run_dir(run_id) / "worker.log"
            if log.exists():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 4000))
                    record["log_tail"] = stream.read().decode("utf-8", errors="replace")
            return record

    def wait(self, run_id: str, wait_seconds: int = 30) -> dict:
        """Bounded long-poll: one model turn can wait without repeated get calls."""
        deadline = time.monotonic() + wait_seconds
        while True:
            with self.store.lock:
                record = self._get_locked(run_id)
            if record["state"] in TERMINAL or record["state"] == "created" or time.monotonic() >= deadline:
                return self.get(run_id)
            time.sleep(min(0.25, max(0, deadline - time.monotonic())))

    def _validate_final(self, record: dict) -> dict:
        folder = self.run_dir(record["run_id"]) / "final-test"
        hashes = tree_hashes(folder)
        if digest(hashes) != record.get("final_hash"):
            raise WorkbenchError("INTEGRITY_MISMATCH", "Final-test artifacts changed")
        result = read_json(folder / "result.json")
        tracked = self.client.get_run(record["run_id"])
        if any(tracked.data.metrics.get("final_test." + key) != value for key, value in result["metrics"].items()):
            raise WorkbenchError("TRACKING_MISMATCH", "MLflow final-test metrics changed")
        artifact_root = Path(url2pathname(urlsplit(tracked.info.artifact_uri).path)).resolve()
        verify_tree(artifact_root / "final-test", hashes)
        return result

    def finalize(self, run_id: str) -> dict:
        """Commit model selection before exposing final labels or scores; same-run retries are safe."""
        with self.store.lock:
            record = self._get_locked(run_id)
            self._validate_outputs(record)
            folder = self.run_dir(run_id)
            split = read_json(folder / "inputs" / "split.json")
            if not split.get("final_test"):
                raise WorkbenchError("NO_FINAL_TEST", "Legacy/opt-out experiment has no reserved final test; create a new experiment")
            contract = self._validate_inputs(record)["contract"]
            key = digest({field: contract[field] for field in ("dataset_sha256", "target", "task")} |
                         {"final_rows": sorted(split["final_test"])})
            selection = self.root / "final-selections" / (key + ".json")
            if selection.exists() and read_json(selection)["run_id"] != run_id:
                raise WorkbenchError("FINAL_TEST_ALREADY_SELECTED", "This final-test partition is committed to another model; do not select models using final-test scores")
            if not selection.exists():
                selection.parent.mkdir(exist_ok=True)
                atomic_json(selection, {"run_id": run_id, "selected_at": time.time(), "partition_key": key})
            if record.get("final_hash"):
                result = self._validate_final(record)
            else:
                import importlib.util
                module = importlib.util.spec_from_file_location("mlwb_final_recipe", folder / "inputs" / "engine.py")
                recipe = importlib.util.module_from_spec(module)
                sys.modules[module.name] = recipe
                # Loading the immutable recipe must not add __pycache__ to its snapshot.
                exec(compile((folder / "inputs" / "engine.py").read_bytes(), str(folder / "inputs" / "engine.py"), "exec"), recipe.__dict__)
                temporary = self.root / "staging" / uuid.uuid4().hex
                try:
                    result = recipe.evaluate_final(folder / "inputs", folder / "output" / "model", temporary)
                    result.update(run_id=run_id, partition_key=key,
                                  note="Model selection was committed before evaluation. This local workflow guard is not access control over the source CSV.")
                    atomic_json(temporary / "result.json", result)
                    if (folder / "final-test").exists():
                        # Recover a crash after artifact commit but before the ledger anchor.
                        if tree_hashes(folder / "final-test") != tree_hashes(temporary):
                            raise WorkbenchError("INTEGRITY_MISMATCH", "Uncommitted final-test output differs from recomputation")
                    else:
                        temporary.rename(folder / "final-test")
                    for name, value in result["metrics"].items():
                        self.client.log_metric(run_id, "final_test." + name, value)
                    self.client.log_artifacts(run_id, str(folder / "final-test"), "final-test")
                    record["final_hash"] = digest(tree_hashes(folder / "final-test"))
                    self.store.save(record)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            return {"run_id": run_id, "state": record["state"], "final_test": result}

    def list(self, limit: int = 20, offset: int = 0, state: str | None = None,
             model: str | None = None, parent_run_id: str | None = None) -> dict:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise WorkbenchError("INVALID_LIMIT", "limit must be between 1 and 100")
        with self.store.lock:
            # Reconcile active records before applying state filters.
            for run_id in self.store.active_ids():
                self._get_locked(run_id)
            ids = self.store.list_ids(limit + 1, offset, state, model, parent_run_id)
            return {"runs": [self._get_locked(run_id) for run_id in ids[:limit]],
                    "next_offset": offset + limit if len(ids) > limit else None}

    def resume(self, previous_session_id: str | None = None, limit: int = 20, offset: int = 0) -> dict:
        request = ResumeRequest(previous_session_id=previous_session_id, limit=limit, offset=offset)
        session_id = request.previous_session_id or (self.dsh_context or {}).get("session_id")
        if session_id is None:
            raise WorkbenchError("DSH_SESSION_REQUIRED", "Call from a DSH Session or supply previous_session_id")
        with self.store.lock:
            links = self.store.session_links(session_id, limit + 1, offset)
            resources = []
            for link in links[:limit]:
                if link["kind"] == "run":
                    record = self._get_locked(link["ref"])
                    resources.append({**link, "state": record["state"], "name": record["name"],
                                      "next_tool": "ml_experiment_get", "arguments": {"run_id": link["ref"]}})
                else:
                    resources.append({**link, "state": "unverified", "next_tool": "ml_search_get",
                                      "arguments": {"study_name": link["ref"]}})
            return {"session_id": session_id, "resources": resources,
                    "next_offset": offset + limit if len(links) > limit else None,
                    "note": "References only; metrics and artifacts are not verified by resume. No training is started. "
                            "A study reference may be an interrupted creation intent; retry its original search_create if absent."}

    def cancel(self, run_id: str) -> dict:
        with self.store.lock:
            record = self._get_locked(run_id)
            if record["state"] in TERMINAL:
                return record
            if record["state"] == "created":
                return self._set_state(record, "cancelled")
            self._set_state(record, "cancelling")
            process = owned_process(record)
        if process is not None:
            try:
                children = process.children(recursive=True)
                for child in children:
                    child.terminate()
                process.terminate()
                _, alive = psutil.wait_procs([*children, process], timeout=3)
                for child in alive:
                    child.kill()
                _, alive = psutil.wait_procs(alive, timeout=3)
                if alive:
                    raise WorkbenchError("CANCEL_PENDING", "Worker termination is still pending; query its state")
            except psutil.NoSuchProcess:
                pass  # The exact owned process already exited; no other PID is targeted.
        with self.store.lock:
            record = self.store.get(run_id)
            if owned_process(record) is not None:
                raise WorkbenchError("CANCEL_PENDING", "Worker is still alive; no cancelled claim was recorded")
            return self._set_state(record, "cancelled")

    def branch(self, run_id: str, changes: dict | None = None, reuse_preprocessing: bool = True,
               request_key: str | None = None) -> dict:
        changes = changes or {}
        forbidden = {"dataset_path", "target", "task", "experiment_name"} & changes.keys()
        if forbidden:
            raise WorkbenchError("INVALID_BRANCH", "Create a new experiment to change dataset, target, task or experiment group")
        payload_hash = digest({"operation": "branch", "run_id": run_id, "changes": changes,
                               "reuse_preprocessing": reuse_preprocessing})
        with self.store.lock:
            previous = self.store.previous_request(request_key, payload_hash)
            if previous is not None:
                return previous
            parent = self._get_locked(run_id)
            self._validate_inputs(parent)
            inputs = self.run_dir(run_id) / "inputs"
            config = read_json(inputs / "config.json")
            config.setdefault("final_test_fraction", 0.0)  # Legacy partitions must not change on branch.
            config["name"] = config["name"] + " branch"
            config["dataset_path"] = str(inputs / "dataset.csv")
            for field in ("preprocessor_path", "estimator_path"):
                if config[field] is not None:
                    config[field] = str(inputs / config[field])
            config.update(changes)
            return self.create(ExperimentSpec.model_validate(config), request_key, parent=parent,
                               reuse_preprocessing=reuse_preprocessing, payload_hash=payload_hash)

    def compare(self, run_ids: list[str], metric: str | None = None) -> dict:
        if not 2 <= len(run_ids) <= 30 or len(set(run_ids)) != len(run_ids):
            raise WorkbenchError("INVALID_COMPARISON", "Provide 2 to 30 distinct run IDs")
        with self.store.lock:
            records = [self._get_locked(run_id) for run_id in run_ids]
            results = [self._validate_outputs(record) for record in records]
            if len({record["comparison_key"] for record in records}) != 1:
                contracts = [read_json(self.run_dir(r["run_id"]) / "manifest.json")["contract"] for r in records]
                keys = set().union(*(contract.keys() for contract in contracts))
                differing = [key for key in sorted(keys) if len({canonical(c.get(key)) for c in contracts}) > 1]
                raise WorkbenchError("INCOMPARABLE_EXPERIMENTS", "Evaluation conditions differ: " + ", ".join(differing))
            metric = metric or ("accuracy" if records[0]["task"] == "classification" else "rmse")
            if metric not in METRICS or any(metric not in result["metrics"] for result in results):
                raise WorkbenchError("INVALID_METRIC", "Metric is not available for every selected run")
            rows = [{"run_id": record["run_id"], "name": record["name"], "model": record["model"],
                     "metrics": result["metrics"], "train_seconds": result["train_seconds"],
                     "reused_preprocessing": record["reused_preprocessing"]}
                    for record, result in zip(records, results, strict=True)]
            rows.sort(key=lambda row: row["metrics"][metric], reverse=METRICS[metric] == "max")
            return {"comparison_key": records[0]["comparison_key"], "metric": metric, "direction": METRICS[metric],
                    "ranking": rows, "note": "Scores use the same validation rows. Repeated selection on this holdout may overfit; timings are single local measurements."}

    def export(self, run_id: str, include_data: bool = False) -> dict:
        with self.store.lock:
            record = self._get_locked(run_id)
            result = self._validate_outputs(record)
            folder = self.run_dir(run_id)
            if record.get("final_hash"):
                result["final_test"] = self._validate_final(record)
            staging = self.root / "staging" / uuid.uuid4().hex
            staging.mkdir(parents=True)
            try:
                shutil.copytree(folder / "inputs", staging / "inputs",
                                ignore=None if include_data else shutil.ignore_patterns("dataset.csv"))
                shutil.copytree(folder / "output", staging / "output")
                if "final_test" in result:
                    shutil.copytree(folder / "final-test", staging / "final-test")
                shutil.copyfile(Path(__file__).with_name("reproduce.py"), staging / "reproduce.py")
                atomic_json(staging / "experiment.json", {"format": 1, "run_id": run_id,
                    "parent_run_id": record["parent_run_id"], "comparison_key": record["comparison_key"],
                    "preprocessing_hash": record["preprocessing_hash"], "metrics": result["metrics"],
                    "final_test": result.get("final_test"), "data_included": include_data, "version": __version__})
                report = [f"# Experiment {run_id}", "", f"Model: {record['model']}",
                          f"Parent: {record['parent_run_id']}", f"Evaluation key: {record['comparison_key']}", "",
                          *[f"- {key}: {value:.12g}" for key, value in result["metrics"].items()], "",
                          "Install inputs/requirements.txt, then run python reproduce.py --verify-only.",
                          "Run python reproduce.py to retrain; pass --data ORIGINAL.csv if data is excluded.",
                          "Only load models and execute factories from a trusted export.", ""]
                report += ["## Independent final test", "", canonical(result.get("final_test")) if "final_test" in result else
                           "Not evaluated. Validation scores must not be described as independent final-test scores.", ""]
                (staging / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
                from .reporting import html_report
                (staging / "REPORT.html").write_text(html_report(record, result,
                    read_json(folder / "inputs" / "dataset.json")), encoding="utf-8")
                atomic_json(staging / "checksums.json", tree_hashes(staging))
                exports = self.root / "exports"
                exports.mkdir(exist_ok=True)
                archive = exports / f"{run_id}-{uuid.uuid4().hex[:8]}.zip"
                temporary = archive.with_suffix(".partial")
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
                    for file in sorted(staging.rglob("*")):
                        if file.is_file():
                            zip_file.write(file, file.relative_to(staging).as_posix())
                temporary.rename(archive)
                return {"run_id": run_id, "path": str(archive), "sha256": sha256(archive),
                        "data_included": include_data, "metrics": result["metrics"], "final_test": result.get("final_test"),
                        "reproduce": "python reproduce.py" + ("" if include_data else " --data ORIGINAL.csv")}
            finally:
                shutil.rmtree(staging)

    def inspect(self, dataset_path: str, target: str | None = None) -> dict:
        return inspect_csv(dataset_path, target)

    def diff(self, left_run_id: str, right_run_id: str) -> dict:
        with self.store.lock:
            records = [self._get_locked(run_id) for run_id in (left_run_id, right_run_id)]
            manifests = [self._validate_inputs(record) for record in records]
            configs = [read_json(self.run_dir(record["run_id"]) / "inputs" / "config.json") for record in records]
            def differences(left, right):
                return {key: {"left": left.get(key), "right": right.get(key)}
                        for key in sorted(left.keys() | right.keys()) if left.get(key) != right.get(key)}
            return {"left_run_id": left_run_id, "right_run_id": right_run_id,
                    "same_evaluation": records[0]["comparison_key"] == records[1]["comparison_key"],
                    "config_changes": differences(*configs),
                    "evaluation_changes": differences(*(m["contract"] for m in manifests)),
                    "code_changed": [name for name in sorted(manifests[0]["files"].keys() | manifests[1]["files"].keys())
                                     if name.endswith(".py") and manifests[0]["files"].get(name) != manifests[1]["files"].get(name)]}

    def predict(self, run_id: str, dataset_path: str) -> dict:
        from .inference import predict_csv
        with self.store.lock:
            record = self._get_locked(run_id)
            self._validate_outputs(record)
            if record["custom_code"] and not self.allow_custom_code:
                raise WorkbenchError("CUSTOM_CODE_DISABLED", "Enable allowCustomCode to load this custom model")
            return predict_csv(self, record, dataset_path)

    def search_create(self, **kwargs) -> dict:
        from .tuning import Tuning
        return Tuning(self).create(**kwargs)

    def search_step(self, study_name: str) -> dict:
        from .tuning import Tuning
        return Tuning(self).step(study_name)

    def search_get(self, study_name: str) -> dict:
        from .tuning import Tuning
        return Tuning(self).get(study_name)

    def search_cancel(self, study_name: str) -> dict:
        from .tuning import Tuning
        return Tuning(self).cancel(study_name)
