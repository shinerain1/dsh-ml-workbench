"""Durable Optuna ask/tell adaptation: one idempotent, nonblocking scheduling step."""

import time
from contextlib import contextmanager

import optuna
from optuna.trial import TrialState
from sqlalchemy.pool import NullPool

from .files import digest, read_json
from .service import METRICS
from .spec import SearchCreateRequest, SearchRequest, WorkbenchError


class Tuning:
    def __init__(self, workbench):
        self.workbench = workbench
        self.uri = "sqlite:///" + (workbench.root / "optuna.sqlite3").as_posix()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    @contextmanager
    def storage(self):
        with self.workbench.store.lock:
            storage = optuna.storages.RDBStorage(self.uri, engine_kwargs={"poolclass": NullPool})
            try:
                yield storage
            finally:
                storage.remove_session()
                storage.engine.dispose()

    def load(self, storage, name):
        SearchRequest(study_name=name)
        try:
            study = optuna.load_study(study_name="mlwb::" + name, storage=storage)
        except KeyError as error:
            raise WorkbenchError("STUDY_NOT_FOUND", "Create this study before querying or stepping it") from error
        if "mlwb.config" not in study.user_attrs:
            raise WorkbenchError("INCOMPLETE_STUDY", "Study creation was interrupted; retry search_create with its original arguments")
        return study

    def create(self, **kwargs):
        request = SearchCreateRequest.model_validate(kwargs)
        with self.storage() as storage:
            record = self.workbench._get_locked(request.run_id)
            manifest = self.workbench._validate_inputs(record)
            config = read_json(self.workbench.run_dir(request.run_id) / "inputs" / "config.json")
            if manifest["contract"]["evaluator"] != "holdout-v2":
                raise WorkbenchError("LEGACY_EVALUATOR", "For tuning, first branch the legacy run with reuse_preprocessing=false")
            if config["model"] == "custom":
                raise WorkbenchError("UNSUPPORTED_SEARCH", "Tuning currently supports built-in sklearn estimators")
            if request.reuse_preprocessing:
                self.workbench._validate_outputs(record)
            metric = request.metric or ("accuracy" if record["task"] == "classification" else "rmse")
            valid = {"accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "average_precision", "log_loss"} if record["task"] == "classification" else {"rmse", "mae", "r2"}
            if metric not in valid:
                raise WorkbenchError("INVALID_METRIC", "Metric does not apply to this task")
            from .engine import build_estimator
            estimator = build_estimator(config, self.workbench.run_dir(request.run_id) / "inputs")
            if set(request.search_space) - estimator.get_params().keys():
                raise WorkbenchError("INVALID_SEARCH_SPACE", "Search parameter is not supported by this estimator")
            payload = {**request.model_dump(), "metric": metric, "base_manifest_hash": record["manifest_hash"],
                       "comparison_key": record["comparison_key"]}
            # Persist a discoverable intent before Optuna commits, even if this control call dies next.
            self.workbench.store.link(self.workbench.dsh_context, "study", request.study_name)
            study = optuna.create_study(study_name="mlwb::" + request.study_name, storage=storage,
                                       direction="maximize" if METRICS[metric] == "max" else "minimize", load_if_exists=True)
            existing = study.user_attrs.get("mlwb.config")
            if existing is not None and digest(existing) != digest(payload):
                raise WorkbenchError("IDEMPOTENCY_CONFLICT", "study_name already has a different immutable configuration")
            if existing is None:
                if study.trials:
                    raise WorkbenchError("INCOMPLETE_STUDY", "Unconfigured study contains trials; use a new study name")
                study.set_user_attr("mlwb.config", payload)
            return self.summary(study)

    @staticmethod
    def key(number):
        return f"mlwb.trial.{number}"

    def prepared_record(self, study, trial):
        mapping = study.user_attrs.get(self.key(trial.number), {})
        run_id = mapping.get("run_id")
        if run_id is not None:
            return mapping, self.workbench._get_locked(run_id)
        request = mapping.get("request")
        if request is not None:
            payload = {"operation": "branch", "run_id": request["run_id"], "changes": request["changes"],
                       "reuse_preprocessing": request["reuse_preprocessing"]}
            record = self.workbench.store.previous_request(request["request_key"], digest(payload))
            if record is not None:
                mapping["run_id"] = record["run_id"]
                study.set_user_attr(self.key(trial.number), mapping)
                return mapping, record
        return mapping, None

    def reconcile(self, study):
        config = study.user_attrs["mlwb.config"]
        for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            mapping, record = self.prepared_record(study, trial)
            if record is None:
                if study.user_attrs.get("mlwb.cancelled"):
                    study.tell(trial.number, state=TrialState.FAIL)
                continue
            expired = (record["state"] in {"starting", "running"} and
                       time.time() - record.get("started_at", time.time()) > config["trial_timeout_seconds"])
            if study.user_attrs.get("mlwb.cancelled") or expired:
                record = self.workbench.cancel(record["run_id"])
                if expired:
                    mapping["failure"] = "Trial timeout observed at polling time"
                    study.set_user_attr(self.key(trial.number), mapping)
            if record["state"] == "succeeded":
                result = self.workbench._validate_outputs(record)
                if config["metric"] not in result["metrics"]:
                    mapping["failure"] = "Requested metric unavailable on these validation labels"
                    study.set_user_attr(self.key(trial.number), mapping)
                    study.tell(trial.number, state=TrialState.FAIL)
                else:
                    study.tell(trial.number, result["metrics"][config["metric"]])
            elif record["state"] in {"failed", "cancelled", "interrupted"}:
                study.tell(trial.number, state=TrialState.FAIL)

    def summary(self, study):
        attrs = study.user_attrs
        config = attrs["mlwb.config"]
        trials = study.get_trials(deepcopy=False)
        active = any(trial.state == TrialState.RUNNING for trial in trials)
        cancelled = attrs.get("mlwb.cancelled", False)
        state = "cancelling" if cancelled and active else "cancelled" if cancelled else (
            "complete" if len(trials) >= config["n_trials"] and not active else "active")
        rows = [{"number": trial.number, "state": trial.state.name, "params": trial.params, "value": trial.value,
                 "run_id": attrs.get(self.key(trial.number), {}).get("run_id"),
                 "failure": attrs.get(self.key(trial.number), {}).get("failure")} for trial in trials]
        best = None
        if any(trial.state == TrialState.COMPLETE for trial in trials):
            trial = study.best_trial
            run_id = attrs[self.key(trial.number)]["run_id"]
            self.workbench._validate_outputs(self.workbench._get_locked(run_id))
            best = {"number": trial.number, "run_id": run_id, "params": trial.params, "value": trial.value}
        return {"study_name": config["study_name"], "state": state, "metric": config["metric"],
                "direction": study.direction.name.lower(), "budget": config["n_trials"], "trials": rows,
                "best": best, "storage_uri": self.uri,
                "next_action": "none" if state in {"cancelled", "complete"} else "Call search_step to reconcile and advance at most one trial",
                "note": "Optimization reuses this validation split; best value is selection-biased. Use independent final-test data. Timeout is enforced on polling, not by a daemon."}

    def get(self, study_name):
        with self.storage() as storage:
            study = self.load(storage, study_name)
            self.reconcile(study)
            return self.summary(study)

    def cancel(self, study_name):
        with self.storage() as storage:
            study = self.load(storage, study_name)
            study.set_user_attr("mlwb.cancelled", True)
            self.reconcile(study)
            return self.summary(study)

    def step(self, study_name):
        with self.storage() as storage:
            study = self.load(storage, study_name)
            config = study.user_attrs["mlwb.config"]
            base = self.workbench._get_locked(config["run_id"])
            self.workbench._validate_inputs(base)
            if base["manifest_hash"] != config["base_manifest_hash"]:
                raise WorkbenchError("INTEGRITY_MISMATCH", "Search base snapshot changed")
            self.reconcile(study)
            if study.user_attrs.get("mlwb.cancelled"):
                return self.summary(study)
            active = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
            if active:
                trial = active[0]
                mapping, record = self.prepared_record(study, trial)
                if not mapping.get("request"):
                    # Crashed during ask/sampling, before any fit could have been started.
                    study.tell(trial.number, state=TrialState.FAIL)
                    return self.summary(study)
                self.launch(study, trial.number, mapping, record)
                return self.summary(study)
            if len(study.trials) >= config["n_trials"]:
                return self.summary(study)
            # Reseed each step from trial count, so restarting the controller does not
            # reset a sampler's sequence to its first suggestion.
            study = optuna.load_study(study_name=study.study_name, storage=storage,
                sampler=optuna.samplers.TPESampler(seed=(config["seed"] + len(study.trials)) % 2**32))
            trial = study.ask()
            try:
                params = dict(read_json(self.workbench.run_dir(config["run_id"]) / "inputs" / "config.json")["params"])
                for name, space in config["search_space"].items():
                    if space["type"] == "categorical":
                        params[name] = trial.suggest_categorical(name, space["choices"])
                    else:
                        options = {"log": space["log"]}
                        if space["step"] is not None:
                            options["step"] = space["step"]
                        function = trial.suggest_int if space["type"] == "int" else trial.suggest_float
                        params[name] = function(name, space["low"], space["high"], **options)
                mapping = {"request": {"run_id": config["run_id"],
                    "changes": {"name": f"{study_name} trial {trial.number}", "params": params},
                    "reuse_preprocessing": config["reuse_preprocessing"],
                    "request_key": f"optuna:{study_name}:{trial.number}"}}
                study.set_user_attr(self.key(trial.number), mapping)
            except Exception:
                study.tell(trial.number, state=TrialState.FAIL)
                raise
            self.launch(study, trial.number, mapping, None)
            return self.summary(study)

    def launch(self, study, number, mapping, record):
        if record is None:
            record = self.workbench.branch(**mapping["request"])
            mapping["run_id"] = record["run_id"]
            study.set_user_attr(self.key(number), mapping)
        if record["comparison_key"] != study.user_attrs["mlwb.config"]["comparison_key"]:
            raise WorkbenchError("INCOMPARABLE_EXPERIMENTS", "Trial evaluator differs from the search base; create a new study after upgrading")
        self.workbench.client.set_tag(record["run_id"], "optuna.study", study.user_attrs["mlwb.config"]["study_name"])
        self.workbench.client.set_tag(record["run_id"], "optuna.trial", str(number))
        self.workbench.start(record["run_id"])
