# 0.1.0 — MLflow workflows adapted for DSH

DSH ML Workbench binds each local CSV experiment to its data, code, parameters, validation rows, MLflow record and fitted pipeline. A follow-up can reuse a prior experiment's exact fitted preprocessing while changing its estimator, without overwriting the original run. Incompatible evaluations and mismatched deliveries are rejected.

Nine DSH tools cover health, create, start, get, list, cancel, branch, compare and export. Python workers persist independently of the DSH turn, expose durable state and confirm cancellation. Exports contain a model, recipe, report, pinned dependencies and standalone reproduction; data is excluded by default.

MLflow 3.18.0 and scikit-learn supply tracking, pipelines, training algorithms, metrics and model serialization. Optuna, DVC and RD-Agent were reviewed but are not runtime integrations in this release.

Validation: 13 Python cases and 3 Node cases; actual DSH tool executor and plugin-manager installation; seven real synthetic-data runs; two exported models independently retrained with matching metrics. See [validation evidence](https://github.com/shinerain1/dsh-ml-workbench/blob/main/docs/VALIDATION.md) for exact scope and clean-environment results.

An online DSH account-provider call returned ACCOUNT_SIGN_IN_REQUIRED before inference. Human intervention reduction and native-versus-plugin agent gains remain unmeasured. The release targets local single-user CSV classification/regression, not time-series or grouped evaluation, multi-tenant service, automatic tuning, or mid-training checkpoint resume.

Install the Python wheel with the supplied requirements.lock, then the DSH `.tgz`, or add this third-party index to DSH Discover:

https://raw.githubusercontent.com/shinerain1/dsh-ml-workbench/main/market/index.json

Configure absolute `pythonExecutable` and `root` paths as shown in the README. Only enable custom factories for reviewed local Python. This is an independent community plugin/index, not an official approval claim.
