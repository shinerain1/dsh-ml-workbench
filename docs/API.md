# Version 1 tool and backend contract

The public DSH tools use ordinary JSON objects; no JSON-encoded strings inside arguments are required. Backend validation is strict: unexpected fields and invalid model/task combinations are rejected. Machine-readable request schemas are in [api-schema.json](api-schema.json); DSH definitions are in [src/index.ts](../src/index.ts).

| DSH tool | RPC operation | Arguments |
|---|---|---|
| ml_workbench_health | health | `{}` |
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

Create optional fields: `name`, `experiment_name`, `model`, `params`, `seed=42`, `test_fraction=0.2` (strictly between 0 and 0.5), `preprocessing`, `preprocessor_path`, `estimator_path`, `request_key`. Supported shared models: `random_forest`, `extra_trees`, `hist_gradient_boosting`. Classification adds `logistic_regression`; regression adds `ridge`. `custom` requires `estimator_path`. Estimator arguments are native scikit-learn parameters; `random_state` and `n_jobs` may not be overridden (use `seed`; n_jobs=1).

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
