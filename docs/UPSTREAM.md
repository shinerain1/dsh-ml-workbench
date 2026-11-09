# Upstream reuse and attribution

The contribution is a DSH adapter for established machine-learning tooling, not new learning algorithms or a replacement experiment tracker. These upstream repositories and their official documentation were reviewed before implementing the adapter.

| Project | Role in 0.1.0 | Reused or informed component |
|---|---|---|
| [MLflow](https://github.com/mlflow/mlflow) (Apache-2.0) | **Runtime dependency**, pinned 3.18.0 | Native `MlflowClient` experiment/run APIs, parameters, metrics, parent-run tags, artifacts, SQLite tracking, `mlflow.sklearn.save_model/load_model`, MLmodel flavor and signature |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) (BSD-3-Clause) | **Runtime dependency** | Estimators, ColumnTransformer, Pipeline, imputers, encoders, scalers, train_test_split and classification/regression metrics |
| [Optuna](https://github.com/optuna/optuna) (MIT) | Reviewed; **not integrated** | Study/trial lifecycle and storage suggest a future mapping from a study to child MLflow runs; adding a dependency without a supported tuning workflow was avoided |
| [DVC](https://github.com/treeverse/dvc) (Apache-2.0) | Reviewed; **not integrated** | Data/parameter/pipeline versioning informed immutable input manifests. Version 0.1.0 uses local copied CSV snapshots, not a DVC remote or cache implementation |
| [RD-Agent](https://github.com/microsoft/RD-Agent) (MIT) | Reviewed; **not integrated** | Research/development iteration informed the explicit propose–run–evaluate–branch workflow. DSH retains the agent role; no RD-Agent orchestration code is bundled |
| [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) | Plugin host and SDK | Cordis `apply`/dependency injection/disposal, `defineTool`, JSON parameter schemas, generic tool cards, bundle patch, plugin-index format |

The projects above are not all DSH plugins. MLflow is the actual tracking backend; Optuna, DVC and RD-Agent are explicitly references/future extensions. No performance claim is inferred from their popularity.

Implementation mapping:

- `python/dsh_ml_workbench/service.py`: MLflow run creation/logging/artifact linkage; adapter snapshots and comparison policy.
- `python/dsh_ml_workbench/engine.py`: standard scikit-learn processing and estimators; MLflow model serialization.
- `python/dsh_ml_workbench/store.py`: a small *process lifecycle* ledger, not a replacement MLflow metric/artifact database.
- `src/index.ts`, `src/client.ts`: DSH tool registration and the Python JSON protocol.

Official references consulted:

- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/), [MlflowClient API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html).
- [scikit-learn: common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), especially fitting preprocessing only on training data.
- [DSH tool authoring](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cookbook/adding-a-tool.md).
- [DSH plugin index guide](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/plugin-market.md).

The adapter has its own MIT license. Upstream packages keep their own licenses; installing them does not transfer their authorship to this project. Dependencies are installed from their distributions, not copied or relicensed here.
