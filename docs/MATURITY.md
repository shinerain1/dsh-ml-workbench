# Capability review: what this adapter does and does not replace

This project is a local DSH integration of mature libraries. Version 0.2.0 is a more capable, tested experiment workflow; it is **not feature-equivalent** to all of MLflow, Optuna, DVC or RD-Agent, and no short local test can establish their level of production maturity.

| Area | v0.1.0 | v0.2.0 | Remaining boundary |
|---|---|---|---|
| Experiment tracking | Native MLflow runs/artifacts | Same, plus Optuna trial links, filters/pagination and config/evaluator/code diff | No multi-user/remote tracking deployment or registry lifecycle adapter |
| Evaluation | Stratified/random IID holdout | Adds group-disjoint and chronological holdouts, probability metrics, dummy baseline and confusion matrix | No nested cross-validation, time-series forecasting framework, embargo/gap or automatic leakage detection |
| Hyperparameter search | Not implemented | Real Optuna 5 TPE ask/tell + SQLite persistence, fixed budget, one active trial, crash reconciliation and cancellation | DSH must keep calling step; no unattended study daemon, parallel scheduling, pruning, multi-objective search or optimizer checkpoints |
| Data/model lineage | Immutable local CSV/code/env snapshots | Same, stricter recipe/parser/library comparison contract; legacy recipe preserved for fitted-preprocessing branches | No DVC remote, content-addressed deduplication, large-data pipeline cache or signed audit trail |
| Delivery | Model/code/Markdown report/standalone reproduction | Adds escaped standalone HTML report and schema-checked batch prediction | No model serving service, canary deployment or GPU inference |
| Agent evaluation | Deterministic integration and failure injection | Extended workflow and upgrade tests | Online DSH natural-language control trial, human-intervention reduction and business gains remain unmeasured |

The first improvement priority was evaluation and failure behavior, not matching a large project's feature count. A repeated customer cannot cross the configured group boundary. Equal timestamps cannot straddle a chronological boundary. A failed MLflow terminal write cannot leave a persisted successful lifecycle claim. A restarted Optuna controller cannot silently launch the same linked fit again.

Official sources reviewed for the gap analysis:

- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/): run/artifact tracking and programmatic/UI comparison; the adapter reuses its tracking backend.
- [Optuna Study API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html) and [ask/tell workflow](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html): the adapter uses create/load study, ask, suggest, tell and persistent RDB storage.
- [DVC experiment comparison](https://doc.dvc.org/user-guide/experiment-management/comparing-experiments): parameter/dependency comparisons informed the DSH diff interface. DVC itself is still not a runtime dependency.
- [RD-Agent](https://github.com/microsoft/RD-Agent): autonomous research/development is much broader than this plugin. DSH continues to provide agent reasoning and code authoring.
- [scikit-learn evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html) and [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html): evaluation must reflect grouped or temporal data, and repeated model selection requires independent final assessment.

Supported deployment: one trusted local user, UTF-8 CSV up to 128 MiB, bounded sequential CPU fits, Windows verified. Raw CSVs, fitted Python models and custom factories remain local. Custom code must be reviewed and explicitly enabled. Cross-platform and large-scale performance have not been established.
