# Version 1 tool and backend contract

The public DSH tools use ordinary JSON objects; no JSON-encoded strings inside arguments are required. Backend validation is strict: unexpected fields and invalid model/task combinations are rejected. Machine-readable request schemas are in [api-schema.json](api-schema.json); DSH definitions are in [src/index.ts](../src/index.ts).

| DSH tool | RPC operation | Arguments |
|---|---|---|
| ml_workbench_health | health | `{}` |
| ml_workbench_resume | resume | optional `previous_session_id`, `limit=20`, `offset=0` |
| ml_experiment_create | create | `dataset_path`, `target`, `task`; optional fields below |
| ml_experiment_start | start | `run_id` |
| ml_experiment_get | get | `run_id` |
| ml_experiment_list | list | `limit` (1–100, default 20) |
| ml_experiment_cancel | cancel | `run_id` |
| ml_experiment_branch | branch | `run_id`, `changes={}`, `reuse_preprocessing=true`, optional `request_key` |
| ml_experiment_compare | compare | 2–30 distinct `run_ids`, optional `metric` |
| ml_experiment_export | export | `run_id`, `include_data=false` |
| ml_dataset_inspect | inspect | `dataset_path`, optional `target` |
| ml_experiment_diff | diff | `left_run_id`, `right_run_id` |
| ml_model_predict | predict | `run_id`, `dataset_path` |
| ml_search_create | search_create | `study_name`, `run_id`, `search_space`, optional budget/metric/seed/reuse/timeout |
| ml_search_step | search_step | `study_name` |
| ml_search_get | search_get | `study_name` |
| ml_search_cancel | search_cancel | `study_name` |

Create requires a UTF-8 CSV up to 128 MiB, at least 10 rows, unique nonempty headers, a known target and at least one feature. Missing labels, infinite numeric values and unsupported labels are rejected. Classification requires two or more discrete classes and a feasible stratified holdout; regression requires a numeric target.

Create optional fields: `name`, `experiment_name`, `model`, `params`, `seed=42`, `test_fraction=0.2` (strictly between 0 and 0.5), `preprocessing`, `preprocessor_path`, `estimator_path`, `request_key`. Supported shared models: `random_forest`, `extra_trees`, `hist_gradient_boosting`. Classification adds `logistic_regression`; regression adds `ridge`. `custom` requires `estimator_path`. Estimator arguments are native scikit-learn parameters. In v0.3 a redundant integer `params.random_state` equal to `seed` and integer `params.n_jobs=1` are accepted and removed from canonical params. Conflicting values, nulls, booleans and floats are rejected. The managed seed and one-thread policy are unchanged. Search spaces cannot tune these controls.

In v0.3 the adapter supplies an optional RPC-envelope `dsh_context: {session_id, call_id}` from DSH execution identity. It is not a tool argument. Create/branch automatically deduplicate exact call replay when no explicit `request_key` is given. New calls and PTC reruns still require a stable explicit key for the same intended experiment. `resume` returns linked resources and their next tools, without training or verifying result metrics. [Complete DSH integration contract](DSH-INTEGRATION.md).

v0.2 adds `split: {strategy:"random"|"group"|"time", column?:string}`. Random omits `column`; group/time require it. Group uses sklearn GroupShuffleSplit (test_fraction is a fraction of groups, not rows). Time requires nonmissing ISO strings; all timestamps at or after the boundary go to validation, so ties may increase its size. The split column is excluded from features. Training must contain all target classes; validation need not, so some probability metrics may be unavailable. Both partitions need at least two rows.

List also accepts `offset=0`, `state`, `model` and `parent_run_id`; it returns `next_offset` or null. Active records are reconciled before state filtering. Classification metrics add `roc_auc`, `average_precision` (binary only) and `log_loss` when probabilities/labels permit them. `classes` gives probability-column order; binary positive class is `classes[1]`, also returned as `positive_class`. Average precision is not an automatically selected business threshold.

Inspect returns aggregate column diagnostics and warnings without raw row previews. Diff returns `config_changes`, `code_changed`, `evaluation_changes` and `same_evaluation`. Prediction requires a successful verified run; missing features and nonnumeric values for numeric features are refused. Extra columns are ignored and reported. It writes a new local input snapshot, predictions CSV and manifest linking input/model/output hashes. Custom models require `allowCustomCode`; inference is a control request, not a persistent training worker.

Search uses real Optuna storage and a finite trial budget. It supports built-in estimators, integer/float/log/categorical parameter distributions and the base run's fixed split. Read the exact [scheduling, timeout and recovery contract](TUNING.md) before treating it as unattended automation. There are four separate operations so an Agent can create, advance, inspect and cancel explicitly.

`preprocessing` supports `numeric_imputer` (median/mean/most_frequent), `scale_numeric=true`, `max_categories=64` and `drop_columns=[]`. Categorical columns use most-frequent imputation and one-hot encoding with unknown categories ignored. Preprocessors are fitted on training rows only.

Factory interfaces:

```python
def make_preprocessor(X_train, config):
    # Return an UNFITTED sklearn transformer; fit is controlled by the worker.
    ...

def make_estimator(task, params, seed):
    # Return an UNFITTED sklearn-compatible classifier or regressor.
    ...
```

Factories must be standalone `.py` files (max 1 MiB); their imported external packages must be installed in the backend environment. Additional local modules/files are not automatically snapshotted. Core packages are version-checked during reproduction; custom code remains responsible for controlling its external resources and randomness.

Backend invocation (one request then process exits):

```sh
python -m dsh_ml_workbench rpc --root /absolute/path/to/records
```

Standard input:

```json
{"protocol_version":1,"operation":"list","args":{"limit":5}}
```

Success stdout:

```json
{"protocol_version":1,"ok":true,"result":{"runs":[]}}
```

Error stdout:

```json
{"protocol_version":1,"ok":false,"error":{"code":"INCOMPARABLE_EXPERIMENTS","message":"Evaluation conditions differ: split_sha256"}}
```

Domain errors are JSON responses with process exit code 0; launch/transport failures may use a nonzero exit code. Diagnostics go to stderr. The Node adapter converts backend errors into failed DSH tool calls. Default request timeout is 120 seconds, with 1 MiB request and 4 MiB response limits; a timeout is not permission to assume a create/start failed. Retry create/branch with the same request key or inspect/list the run first.

Create/branch returns a record with `run_id`, state, model, task, parent ID and integrity fingerprints. `get` adds lifecycle events and a bounded log tail, plus metrics/model URI after verified success. Compare returns the shared comparison key, metric/direction and ranked runs. Export returns absolute local ZIP path, SHA-256, metrics, data-inclusion flag and reproduction command; it performs no external upload.

Error codes include `INVALID_ARGUMENTS`, `INVALID_DATASET`, `INVALID_TARGET`, `INVALID_FEATURES`, `CUSTOM_CODE_DISABLED`, `IDEMPOTENCY_CONFLICT`, `RUN_NOT_FOUND`, `INTEGRITY_MISMATCH`, `TRACKING_MISMATCH`, `PREPROCESSING_MISMATCH`, `INCOMPARABLE_EXPERIMENTS`, `LAUNCH_FAILED`, `PROCESS_STATE_UNKNOWN` and `CANCEL_PENDING`. Treat other codes as failures and inspect the message. No automatic repeated fit is implied.

New errors include `INVALID_SPLIT`, `FEATURE_SCHEMA_MISMATCH`, `STUDY_NOT_FOUND`, `INCOMPLETE_STUDY`, `INVALID_SEARCH_SPACE`, `UNSUPPORTED_SEARCH` and `LEGACY_EVALUATOR`. A failed or abandoned Optuna trial counts against its budget; a study with all failures has `best:null` even when `state:complete`.

## v0.4 additions and compatibility

| DSH tool | RPC operation | Inputs |
|---|---|---|
| ml_experiment_run | run | Same as create; snapshots and starts in one call |
| ml_experiment_wait | wait | run_id, wait_seconds=30 (0-60) |
| ml_experiment_finalize | finalize | run_id; commits model choice before final-test evaluation |

Create adds final_test_fraction=0.2 (0 <= value < 0.5) and timeout_seconds=3600 (integer 5-86400). test_fraction remains the validation fraction with its legacy name. Their sum must be below 0.8. Split first reserves the final partition, then splits development data using validation_fraction/(1-final_fraction). Random uses sklearn stratification; group uses fractions of distinct groups; time preserves timestamp ties. Each nonempty partition needs at least two rows. Train must contain all classes. Old snapshots lacking final_test_fraction branch with 0.0, preserving their partition and frozen evaluator; they cannot finalize.

Normal train/search/compare outputs contain validation metrics only. Finalize never refits; it uses the frozen evaluator and fitted model, writes separately hashed final-test artifacts, and logs MLflow metrics under final_test.*. Identical final rows with the same data/target/task are locked to the chosen run even when validation configuration or evaluator differs. A failed final evaluation can be retried only for that chosen run. Export includes final metrics only after finalization, and reproduce.py refits a new estimator and compares both sets separately. Timings are not reproduction equality targets.

expected_backend_version is an optional RPC envelope field; the DSH adapter always supplies 0.4.0. --external-workers is a host adapter option, never a model argument. DSH mode consumes versioned launch descriptors internally; standalone callers should use normal local mode. doctor is a separate CLI command, not an RPC operation. The protocol remains version 1 with additive fields.
