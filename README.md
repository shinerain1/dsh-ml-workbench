# DSH ML Workbench

[中文说明](README.zh.md) · [API](docs/API.md) · [Validation](docs/VALIDATION-v0.2.md) · [Upstream reuse](docs/UPSTREAM.md)

A DeepSeek Harness plugin adapting **MLflow, Optuna and scikit-learn** into an agent-accessible experiment workflow. Version 0.2.0 provides 16 tools for CSV inspection, experiments, durable tuning, comparisons, prediction and reproducible delivery. Training algorithms, sampling, model serialization and tracking come from established libraries. See the candid [capability review](docs/MATURITY.md) for the remaining gap to full upstream platforms.

Ask DSH to train several repurchase classifiers, then say: “Keep the third experiment's fitted preprocessing, try a faster model, and compare.” The branch gets a new MLflow run, copies the exact fitted transformer, retains the same validation rows and leaves the parent untouched.

## Install

Requires Python 3.11+ (the lock file was tested on 3.12.0), Node 24.9+, and DSH 0.2.2 (tested against `0.2.2-alpha.3`). This is a local, single-user backend. Install the Python backend separately from the DSH bundle.

Download the Python wheel, `requirements.lock` and plugin `.tgz` from [Releases](https://github.com/shinerain1/dsh-ml-workbench/releases). Initialize DSH Desktop once, then fully quit it before installing into its profile. On Windows, from a dedicated folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock .\dsh_ml_workbench-0.2.0-py3-none-any.whl
dsh plugin --profile desktop add .\dsh-ml-workbench-0.2.0.tgz
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

1. `ml_dataset_inspect` reports missing values, types, duplicates and possible identifiers. `ml_experiment_create` then snapshots CSV, code, parameters, environment and validation indices. A `request_key` makes creation retries idempotent.
2. `ml_experiment_start` starts one detached worker. Poll `ml_experiment_get`; the control call returns before training completes. Closing DSH does not cancel the worker.
3. `ml_experiment_list` restores run IDs after a session restart. A dead worker is reported as `interrupted`, never successful.
4. `ml_experiment_branch` creates a new experiment. By default it reuses a successful parent's exact fitted preprocessing. Set `reuse_preprocessing: false` to change processing or restart a failed fit.
5. `ml_experiment_diff` explains configuration, code and evaluator changes. `ml_experiment_compare` refuses incompatible conditions. Classification adds ROC-AUC, average precision and log loss when defined; reports include a dummy baseline and confusion matrix.
6. `ml_search_create/step/get/cancel` adapt persistent Optuna tuning. Each step starts at most one asynchronous trial; repeating calls is safe, and the same study resumes after a DSH restart. [Search contract](docs/TUNING.md).
7. `ml_model_predict` applies a verified model to another local CSV, checks required features and writes predictions/probabilities plus a manifest.
8. `ml_experiment_export` verifies snapshots and MLflow metrics before delivering a ZIP with model, code, HTML/Markdown report, environment and standalone reproduction. CSV is omitted unless `include_data: true`.

Example creation arguments:

```json
{"dataset_path":"C:/data/customers.csv","target":"repurchased","task":"classification","name":"forest","model":"random_forest","params":{"n_estimators":100},"preprocessing":{"drop_columns":["customer_id"]},"request_key":"customers-forest-001"}
```

Example branch arguments:

```json
{"run_id":"<third run ID>","changes":{"name":"third preprocessing, linear model","model":"logistic_regression","params":{}},"reuse_preprocessing":true,"request_key":"customers-linear-004"}
```

`changes` replaces specified top-level fields, including the whole `params` object. No run ID is reused for a changed experiment. Branching is restart-from-data, not optimizer checkpoint continuation.

Choose evaluation boundaries explicitly. `split: {"strategy":"group","column":"customer_id"}` keeps a customer entirely on one side; `split: {"strategy":"time","column":"event_date"}` puts later ISO timestamps in validation without splitting equal timestamps. The split column is automatically excluded from features. Time mode does not implement a forecasting pipeline or an embargo gap; grouped holdout does not guarantee class balance.

When upgrading from 0.1.0, keep the same root. Old records remain readable/exportable. A fitted-preprocessing branch retains the parent's snapshotted recipe and evaluator; cross-version evaluators are not silently ranked together. To use new metrics or tuning, create a branch with `reuse_preprocessing:false` to refit using the current recipe.

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

The default is an IID holdout, with explicit group/time alternatives. Forecasting, GPU training, nested cross-validation, automatic leakage detection, DVC storage, model serving and RD-Agent orchestration are not implemented. Optuna runs sequentially; DSH must poll step to advance, and timeout is enforced on polling, not by a background daemon. Remove identifiers and post-outcome columns explicitly. Repeated model selection can overfit the holdout; evaluate the final chosen approach independently. Local synthetic results do not establish business value or improved agent productivity.

## Development and evidence

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
npm ci --ignore-scripts
npm run check
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts/demo.py --root .verification/new-demo --report docs/validation-results.json
.\.venv\Scripts\python.exe scripts/demo-v2.py --root .verification/new-demo-v2 --report docs/v2-results.json
```

Use a new demo root each time. [Validation](docs/VALIDATION-v0.2.md) reports actual runs, failure injection and limitations. [Architecture](docs/ARCHITECTURE.md) explains which guarantees the adapter adds. MIT license for this adapter; upstream licenses remain with their projects.
