"""Agent workflow around MLflow, with immutable inputs and explicit worker ownership."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from . import __version__
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
from .spec import ExperimentSpec, WorkbenchError
from .store import TERMINAL, Store

MAX_DATASET_BYTES = 128 * 1024 * 1024
STARTUP_GRACE_SECONDS = 30
METRICS = {"accuracy": "max", "balanced_accuracy": "max", "f1_macro": "max",
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
    def __init__(self, root: Path, *, allow_custom_code: bool = False):
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
        self.store.save(record)
        self.client.set_tag(record["run_id"], "mlwb.state", state)
        if state in TERMINAL:
            status = "FINISHED" if state == "succeeded" else "KILLED" if state == "cancelled" else "FAILED"
            self.client.set_terminated(record["run_id"], status)
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
        return result

    def create(self, spec: ExperimentSpec, request_key: str | None = None, *, parent: dict | None = None,
               reuse_preprocessing: bool = False, payload_hash: str | None = None) -> dict:
        payload_hash = payload_hash or digest({"operation": "create", "spec": spec.model_dump()})
        with self.store.lock:
            previous = self.store.previous_request(request_key, payload_hash)
            if previous is not None:
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
                with (inputs / "dataset.csv").open(encoding="utf-8-sig", newline="") as stream:
                    header = next(csv.reader(stream), [])
                if not header or len(set(header)) != len(header) or any(not value for value in header):
                    raise WorkbenchError("INVALID_DATASET", "CSV header must contain unique, nonempty column names")
                frame = pd.read_csv(inputs / "dataset.csv", low_memory=False)
                if spec.target not in frame or len(frame) < 10:
                    raise WorkbenchError("INVALID_DATASET", "Target must exist and the CSV must contain at least 10 rows")
                if frame[spec.target].isna().any():
                    raise WorkbenchError("INVALID_TARGET", "Target contains missing values; clean labels explicitly")
                missing = set(spec.preprocessing.drop_columns) - set(frame.columns)
                if missing or len(set(frame.columns) - {spec.target, *spec.preprocessing.drop_columns}) == 0:
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
                train, test = train_test_split(np.arange(len(frame)), test_size=spec.test_fraction,
                                              random_state=spec.seed,
                                              stratify=labels if spec.task == "classification" else None)
                if len(test) < 2:
                    raise WorkbenchError("INVALID_SPLIT", "Validation set must contain at least two rows")
                split = {"train": train.tolist(), "test": test.tolist()}
                atomic_json(inputs / "split.json", split)
                dataset_hash = sha256(inputs / "dataset.csv")
                atomic_json(inputs / "dataset.json", {"sha256": dataset_hash, "rows": len(frame),
                                                     "columns": frame.columns.tolist(), "parser": "pandas-csv-v1"})
                portable = spec.model_dump()
                portable["dataset_path"] = "dataset.csv"
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
                contract = {"dataset_sha256": dataset_hash, "split_sha256": digest(split),
                            "target": spec.target, "task": spec.task, "evaluator": "holdout-v1",
                            "metric_library": env["packages"]["scikit-learn"], "parser": "pandas-csv-v1"}
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
                atomic_json(staging / "manifest.json", manifest)
                experiment = self.client.get_experiment_by_name(spec.experiment_name)
                experiment_id = experiment.experiment_id if experiment else self.client.create_experiment(
                    spec.experiment_name, artifact_location=(self.root / "mlflow-artifacts").as_uri())
                tags = {"mlwb.version": __version__, "mlwb.state": "created", "mlwb.comparison_key": digest(contract),
                        "mlwb.dataset_sha256": dataset_hash, "mlwb.preprocessing_hash": preprocessor_hash}
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

    def start(self, run_id: str) -> dict:
        with self.store.lock:
            record = self._get_locked(run_id)
            if record["state"] != "created":
                return record
            self._validate_inputs(record)
            if record["custom_code"] and not self.allow_custom_code:
                raise WorkbenchError("CUSTOM_CODE_DISABLED", "Enable allowCustomCode before starting this experiment")
            token = uuid.uuid4().hex
            record.update(launch_token=token, started_at=time.time())
            self._set_state(record, "starting")
            argv = [sys.executable, "-m", "dsh_ml_workbench", "worker", "--root", str(self.root),
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
            record["events"] = self.store.events(run_id)
            log = self.run_dir(run_id) / "worker.log"
            if log.exists():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 4000))
                    record["log_tail"] = stream.read().decode("utf-8", errors="replace")
            return record

    def list(self, limit: int = 20) -> dict:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise WorkbenchError("INVALID_LIMIT", "limit must be between 1 and 100")
        with self.store.lock:
            return {"runs": [self._get_locked(run_id) for run_id in self.store.list_ids(limit)]}

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
                differing = [key for key in contracts[0] if len({canonical(c[key]) for c in contracts}) > 1]
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
            staging = self.root / "staging" / uuid.uuid4().hex
            staging.mkdir(parents=True)
            try:
                shutil.copytree(folder / "inputs", staging / "inputs",
                                ignore=None if include_data else shutil.ignore_patterns("dataset.csv"))
                shutil.copytree(folder / "output", staging / "output")
                shutil.copyfile(Path(__file__).with_name("reproduce.py"), staging / "reproduce.py")
                atomic_json(staging / "experiment.json", {"format": 1, "run_id": run_id,
                    "parent_run_id": record["parent_run_id"], "comparison_key": record["comparison_key"],
                    "preprocessing_hash": record["preprocessing_hash"], "metrics": result["metrics"],
                    "data_included": include_data, "version": __version__})
                report = [f"# Experiment {run_id}", "", f"Model: {record['model']}",
                          f"Parent: {record['parent_run_id']}", f"Evaluation key: {record['comparison_key']}", "",
                          *[f"- {key}: {value:.12g}" for key, value in result["metrics"].items()], "",
                          "Install inputs/requirements.txt, then run python reproduce.py --verify-only.",
                          "Run python reproduce.py to retrain; pass --data ORIGINAL.csv if data is excluded.",
                          "Only load models and execute factories from a trusted export.", ""]
                (staging / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
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
                        "data_included": include_data, "metrics": result["metrics"],
                        "reproduce": "python reproduce.py" + ("" if include_data else " --data ORIGINAL.csv")}
            finally:
                shutil.rmtree(staging)
