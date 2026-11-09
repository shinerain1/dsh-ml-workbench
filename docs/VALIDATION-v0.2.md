# Version 0.2.0 validation

The [capability review](MATURITY.md) distinguishes implemented behavior from full upstream feature parity. This report is about executed local checks, not a claim of production maturity or improved agent productivity.

## Regression and failure injection

[test-results-v2.json](test-results-v2.json) is generated from pytest JUnit output, removing machine names and absolute paths. It lists each passed case rather than counting repeated runs as additional coverage. The original 20-case regression run passed in 798.29 seconds, with one upstream SQLAlchemy deprecation warning. Two additional targeted cases passed in 137.83 seconds for active-study timeout/cancellation and recovery after a committed branch loses its Optuna run link. A final artifact-copy corruption case passed in 84.57 seconds. Total: **23 distinct passing Python cases**; repeated runs are not added to the count.

Coverage includes real classification/regression fits, serialized predictions, fitted-preprocessor branches, standalone exports, group/time boundaries including timestamp ties, missing features, probability metrics, baseline/confusion reports, escaped HTML labels, study persistence, concurrent step calls, finite budgets, cancellation, lost responses, worker death, PID reuse, input/metric tampering, MLflow artifact-copy corruption and terminal-write failure.

The Node suite passed all **3 cases** (95.66 seconds), driving the Python backend through the official tool registry. The actual DSH checkout `8bd073392eef6f54e921a093b95076248eb08cbb` (`0.2.2-alpha.3`) mounted **16 tools**, executed health through its native batch executor and disposed them successfully. TypeScript build/typecheck and Python lint passed.

## Synthetic workflow

[v2-results.json](v2-results.json) comes from `scripts/demo-v2.py`:

- A grouped classification base plus **3 completed Optuna trials**, each linked to a distinct MLflow run. No configured customer group appears on both sides of the holdout.
- Best validation ROC-AUC **0.9551587302**, for max_depth=3 in this small generated dataset. This is a selected validation value, not an independent test result or evidence that TPE beats another sampler.
- The selected model predicts **20 unlabeled rows**, with class probabilities and input/model/output hashes.
- A chronological regression experiment puts every training timestamp before the first validation timestamp. Ridge RMSE **3.7225867424**; dummy-mean baseline RMSE **84.5297368024** on this synthetic holdout.
- Two local delivery archives are produced for independent reproduction.

The groups and dates are synthetic fixtures designed to test boundary behavior, not a realistic customer simulation or forecasting benchmark. Fit timings are single local measurements.

## Upgrade and packaging

[upgrade-results.json](upgrade-results.json) records training with the actual tagged v0.1.0 Python source, followed by a v0.2.0 fitted-preprocessor branch, comparison and export. The branch preserves the original engine bytes and evaluation identity. It does not silently convert an old score to a new evaluator.

[packaging-results.json](packaging-results.json) records wheel installation into a new virtual environment and standalone reconstruction of the two v0.2 delivery archives. The reproduction helper validates checksums, dependency versions and numerical metrics. Installed source must come from site-packages rather than the editable repository.

Reproduce the checks from a clone:

```powershell
npm ci --ignore-scripts
npm run check
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts/demo-v2.py --root .verification/fresh-v2-demo --report docs/v2-results.json
.\.venv\Scripts\python.exe scripts/verify-upgrade.py
```

Use fresh verification directories. Historical v0.1.0 evidence is available at its immutable [release tag](https://github.com/shinerain1/dsh-ml-workbench/blob/v0.1.0/docs/VALIDATION.md).

## Online-agent limitation

A fresh v0.2 headless DSH health-call attempt again returned **ACCOUNT_SIGN_IN_REQUIRED** before inference. No live natural-language workflow or native-DSH-versus-plugin agent trial is claimed. Manual intervention rates, real customer-model performance and team-scale operations remain unmeasured. Existing credentials were not printed or published.

Optuna scheduling is driven by step calls. An already running fit survives DSH exit; a new trial does not start until another step call. Timeouts are enforced when polled, and custom code is trusted host-local Python. These limits are part of the interface, not hidden by a success status.
