# DSH ML Workbench

[中文说明](README.zh.md) · [API](docs/API.md) · [Validation](docs/VALIDATION-v0.4.md) · [Upstream reuse](docs/UPSTREAM.md)

A DeepSeek Harness plugin adapting **MLflow, Optuna and scikit-learn** into an agent-accessible experiment workflow. Version 0.4.0 provides 20 tools for CSV inspection, experiments, durable tuning, comparisons, prediction and reproducible delivery. Training algorithms, sampling, model serialization and tracking come from established libraries. See the candid [capability review](docs/MATURITY.md) for the remaining gap to full upstream platforms.

Ask DSH to train several repurchase classifiers, then say: “Keep the third experiment's fitted preprocessing, try a faster model, and compare.” The branch gets a new MLflow run, copies the exact fitted transformer, retains the same validation rows and leaves the parent untouched.

## Install

Requires Python 3.11+ (the lock file was tested on 3.12.0), Node 24.9+, and DSH 0.2.2 (tested against `0.2.2-alpha.3`). This is a local, single-user backend. Install the Python backend separately from the DSH bundle.

Download the Python wheel, `requirements.lock` and plugin `.tgz` from [Releases](https://github.com/shinerain1/dsh-ml-workbench/releases). Initialize DSH Desktop once, then fully quit it before installing into its profile. On Windows, from a dedicated folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock .\dsh_ml_workbench-0.4.0-py3-none-any.whl
dsh plugin --profile desktop add .\dsh-ml-workbench-0.4.0.tgz
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
    root: C:/my-project/.dsh-ml-workbench
    executionMode: dsh
    allowCustomCode: false
```

Call `ml_workbench_health` first. Use the same `root` after restarting DSH to recover runs. On macOS/Linux use `.venv/bin/python`; these platforms have not been verified for this release.

Alternatively run the installation helper from this checkout or an extracted plugin `package` directory:

```powershell
py -3.12 scripts/install.py --venv C:/ml-workbench/.venv --workspace C:/my-project --wheel C:/downloads/dsh_ml_workbench-0.4.0-py3-none-any.whl
```

It installs locked dependencies, checks backend/plugin version equality and local SQLite writes, and writes `ml-workbench.config.json` inside the venv. Apply that snippet to the plugin; the helper never overwrites a DSH profile. In `dsh` mode the root must be in the Session workspace or an explicitly approved writable root. No extra grants are added. Run `python -m dsh_ml_workbench doctor --root PATH` independently to diagnose an installation; only an actual DSH tool call verifies OS sandbox access.

## Workflow

1. `ml_dataset_inspect` reports missing values, types, duplicates and possible identifiers. `ml_experiment_create` then snapshots CSV, code, parameters, environment and validation indices. A `request_key` makes creation retries idempotent.
2. `ml_experiment_run` combines create and start; the two original tools remain available. Use `ml_experiment_wait` for bounded waits and [DSH PTC](docs/PTC-WORKFLOW.md) to compose calls. Local workers survive DSH shutdown; DSH-mode workers stop with their owning plugin/process range.
3. `ml_experiment_list` restores run IDs after a session restart. A dead worker is reported as `interrupted`, never successful.
4. `ml_experiment_branch` creates a new experiment. By default it reuses a successful parent's exact fitted preprocessing. Set `reuse_preprocessing: false` to change processing or restart a failed fit.
5. `ml_experiment_diff` explains configuration, code and evaluator changes. `ml_experiment_compare` refuses incompatible conditions. Classification adds ROC-AUC, average precision and log loss when defined; reports include a dummy baseline and confusion matrix.
6. `ml_search_create/step/get/cancel` adapt persistent Optuna tuning. Each step starts at most one asynchronous trial; repeating calls is safe, and the same study resumes after a DSH restart. [Search contract](docs/TUNING.md).
7. `ml_model_predict` applies a verified model to another local CSV, checks required features and writes predictions/probabilities plus a manifest.
8. Select one model using validation only, then call `ml_experiment_finalize` to evaluate its reserved final test. New runs default to 60/20/20 train/validation/final-test partitions (group/time constraints may adjust row counts).
9. `ml_experiment_export` verifies snapshots and MLflow metrics before delivering a ZIP with model, code, HTML/Markdown report, environment and standalone reproduction. CSV is omitted unless `include_data: true`.

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

DSH can author a scikit-learn transformer factory (`examples/preprocessor.py`) or estimator factory (`examples/estimator.py`). Enable `allowCustomCode` only for reviewed local Python. `executionMode: dsh` routes control and training through DSH sandbox, sandboxPolicy and subprocess services with no local fallback. `local` preserves durable host execution. Windows DSH reports partial file confinement: this is not a read restriction, network sandbox or hostile-code boundary. Models use Python serialization; load only trusted artifacts. A root is for one local user, not an untrusted multi-tenant server.

The default is an IID holdout, with explicit group/time alternatives. Forecasting, GPU training, nested cross-validation, automatic leakage detection, DVC storage, model serving and RD-Agent orchestration are not implemented. Optuna runs sequentially; DSH must poll step to advance, while each new fit has an independently enforced background deadline. Remove identifiers and post-outcome columns explicitly. Repeated model selection can overfit validation. Finalize the chosen model once on the reserved final partition; the same partition can select only one run in this local root. This is a workflow guard, not access control over the original CSV. Legacy two-way splits remain unchanged; final_test_fraction=0 explicitly opts out. Local synthetic results do not establish business value or improved agent productivity.

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

Use a new demo root each time. [Validation](docs/VALIDATION-v0.4.md) reports actual runs, failure injection and limitations. [Architecture](docs/ARCHITECTURE.md) explains which guarantees the adapter adds. MIT license for this adapter; upstream licenses remain with their projects.

Session recovery and exact-call retry identity: [DSH integration](docs/DSH-INTEGRATION.md). Controlled comparison: [experiment report](docs/BENCHMARK.md).
