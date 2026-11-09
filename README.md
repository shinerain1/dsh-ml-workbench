# DSH ML Workbench

[中文说明](README.zh.md) · [API](docs/API.md) · [Validation](docs/VALIDATION.md) · [Upstream reuse](docs/UPSTREAM.md)

A DeepSeek Harness plugin adapting **MLflow and scikit-learn** into an agent-accessible experiment workflow. It supports local CSV classification and regression: create, start, inspect, cancel, branch, compare and export. Training algorithms, pipelines, model serialization and experiment tracking come from established libraries.

Ask DSH to train several repurchase classifiers, then say: “Keep the third experiment's fitted preprocessing, try a faster model, and compare.” The branch gets a new MLflow run, copies the exact fitted transformer, retains the same validation rows and leaves the parent untouched.

## Install

Requires Python 3.11+ (tested on 3.12.0), Node 24.9+, and DSH 0.2.2 (tested against `0.2.2-alpha.3`). Version 0.1.0 is a local, single-user backend. Install the Python backend separately from the DSH bundle.

Download the Python wheel, `requirements.lock` and plugin `.tgz` from [Releases](https://github.com/shinerain1/dsh-ml-workbench/releases). Initialize DSH Desktop once, then fully quit it before installing into its profile. On Windows, from a dedicated folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock .\dsh_ml_workbench-0.1.0-py3-none-any.whl
dsh plugin --profile desktop add .\dsh-ml-workbench-0.1.0.tgz
```

Alternatively enable **Custom plugin index** in DSH's experimental plugins, open **Discover**, and add:

```text
https://raw.githubusercontent.com/shinerain1/dsh-ml-workbench/main/market/index.json
```

This is a published third-party index. DSH currently has no bundled central index; this is not a claim of official endorsement. The index pins an immutable Git commit and the Git package includes compiled JavaScript. Python must still be installed as above.

Set the plugin's `pythonExecutable` to the **absolute** venv interpreter path and `root` to an absolute writable data directory. The plugin settings also accept this DSH patch:

```yaml
- id: ml-workbench
  config:
    pythonExecutable: C:/ml-workbench/.venv/Scripts/python.exe
    root: C:/ml-workbench/records
    allowCustomCode: false
```

Call `ml_workbench_health` first. Use the same `root` after restarting DSH to recover runs. On macOS/Linux use `.venv/bin/python`; these platforms have not been verified for this release.

## Workflow

1. `ml_experiment_create` snapshots CSV, code, parameters, environment and validation indices. A `request_key` makes retries of this creation idempotent.
2. `ml_experiment_start` starts one detached worker. Poll `ml_experiment_get`; the control call returns before training completes. Closing DSH does not cancel the worker.
3. `ml_experiment_list` restores run IDs after a session restart. A dead worker is reported as `interrupted`, never successful.
4. `ml_experiment_branch` creates a new experiment. By default it reuses a successful parent's exact fitted preprocessing. Set `reuse_preprocessing: false` to change processing or restart a failed fit.
5. `ml_experiment_compare` refuses different data, targets, splits or evaluator contracts. Supported metrics are accuracy/balanced accuracy/macro-F1 and RMSE/MAE/R².
6. `ml_experiment_export` verifies snapshots and MLflow metrics before delivering a local ZIP. It includes model, code, report, environment and standalone reproduction; CSV is omitted unless `include_data: true`.

Example creation arguments:

```json
{"dataset_path":"C:/data/customers.csv","target":"repurchased","task":"classification","name":"forest","model":"random_forest","params":{"n_estimators":100},"preprocessing":{"drop_columns":["customer_id"]},"request_key":"customers-forest-001"}
```

Example branch arguments:

```json
{"run_id":"<third run ID>","changes":{"name":"third preprocessing, linear model","model":"logistic_regression","params":{}},"reuse_preprocessing":true,"request_key":"customers-linear-004"}
```

`changes` replaces specified top-level fields, including the whole `params` object. No run ID is reused for a changed experiment. Branching is restart-from-data, not optimizer checkpoint continuation.

## Delivery and MLflow

Unzip a delivery into a dedicated directory and use a fresh virtual environment:

```sh
python -m pip install -r inputs/requirements.txt
python reproduce.py --verify-only
python reproduce.py --data /absolute/path/to/original.csv
```

Omit `--data` when the export includes its CSV. Reproduction checks the data hash, environment, snapshot checksums and reproduced metrics. The original workbench package and tracking database are not needed. Load a trusted exported model with `mlflow.sklearn.load_model("output/model")`; its preprocessing is included.

The backend uses native MLflow tracking. `ml_workbench_health` returns the `tracking_uri`. To browse it locally:

```sh
mlflow ui --backend-store-uri sqlite:///C:/ml-workbench/records/mlflow.sqlite3 --host 127.0.0.1
```

Do not edit tracked metrics or snapshot files: consistency validation will reject mismatched deliveries.

## Custom code and scope

DSH can author a scikit-learn transformer factory (`examples/preprocessor.py`) or estimator factory (`examples/estimator.py`). Enable `allowCustomCode` only for reviewed local Python. The plugin runs on the DSH host using Node subprocesses; it does **not** run inside DSH's SSH/process sandbox. Factories execute with the user's permissions and models use Python serialization; load only trusted artifacts. A root is for one local user, not an untrusted multi-tenant server.

The first release uses an IID holdout (stratified for classification). It does not implement time/group-aware splitting, forecasting, GPU training, automatic feature-leakage detection, Optuna tuning, DVC storage or RD-Agent orchestration. Remove identifiers and post-outcome columns explicitly. Repeated selection on one holdout can overfit; evaluate the final chosen approach on independently held-out data. Local synthetic results do not establish customer prediction quality or improved agent productivity.

## Development and evidence

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
npm ci --ignore-scripts
npm run check
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts/demo.py --root .verification/new-demo --report docs/validation-results.json
```

Use a new demo root each time. [Validation](docs/VALIDATION.md) reports actual runs, failure injection and limitations. [Architecture](docs/ARCHITECTURE.md) explains which guarantees the adapter adds. MIT license for this adapter; upstream licenses remain with their projects.
