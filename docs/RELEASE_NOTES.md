# 0.2.0 — Durable Optuna tuning and stronger experiment workflows

DSH ML Workbench binds each local CSV experiment to its data, code, parameters, validation rows, MLflow record and fitted pipeline. A follow-up can reuse a prior experiment's exact fitted preprocessing while changing its estimator, without overwriting the original run. Incompatible evaluations and mismatched deliveries are rejected.

Sixteen DSH tools add dataset inspection, experiment diffs, batch prediction and durable Optuna search to the original experiment lifecycle. Group-disjoint and chronological holdouts, probability metrics, dummy baselines and confusion matrices improve evaluation. Deliveries now include an escaped portable HTML report.

MLflow 3.18.0, Optuna 5.0.0 and scikit-learn provide the underlying tracking, sampling, training and serialization. Optuna studies persist in SQLite with a linked MLflow run per trial. Repeated step calls do not duplicate active fits; lost controller responses are reconciled through persisted request keys. DVC and RD-Agent remain references rather than runtime integrations.

Validation covers the expanded Python regression/failure-injection suite, three Node tests, the actual 16-tool DSH executor, grouped/time synthetic fits, three Optuna trials, batch prediction, legacy v0.1 upgrade and independent delivery reproduction. See [v0.2 validation evidence](https://github.com/shinerain1/dsh-ml-workbench/blob/main/docs/VALIDATION-v0.2.md) and its machine-readable results.

An online DSH account-provider call still returns ACCOUNT_SIGN_IN_REQUIRED before inference. Agent-productivity gains remain unmeasured. This remains a local single-user CSV workflow: Optuna requires step calls to advance and enforces timeout on polling; parallel tuning, pruning, nested CV, GPU training, remote serving and optimizer checkpoint continuation are not implemented. It is not feature-equivalent to the full upstream platforms.

Install the Python wheel with the supplied requirements.lock, then the DSH `.tgz`, or add this third-party index to DSH Discover:

https://raw.githubusercontent.com/shinerain1/dsh-ml-workbench/main/market/index.json

Configure absolute `pythonExecutable` and `root` paths as shown in the README. Only enable custom factories for reviewed local Python. This is an independent community plugin/index, not an official approval claim.
