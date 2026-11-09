# Changelog

## 0.2.0

- Integrate Optuna 5.0.0 TPE ask/tell with durable SQLite studies, MLflow trial links, bounded sequential scheduling, retry-safe recovery and cancellation.
- Add explicit group-disjoint and chronological validation splits; automatically exclude the split column from model features.
- Add dataset diagnostics, experiment config/code/evaluator diff, list filtering/pagination and schema-checked batch prediction.
- Add probability metrics, dummy baseline, confusion matrix and portable HTML delivery reports.
- Strengthen comparison fingerprints with recipe and parser-library versions; persist successful lifecycle state only after MLflow termination succeeds.
- Preserve legacy snapshotted recipe/evaluator when branching with an existing fitted preprocessor.
- Expand failure-injection, upgrade and clean-package validation. Full upstream parity and agent-productivity gains are not claimed.

## 0.1.0

- Initial nine-tool DSH adapter for local MLflow/sklearn classification and regression experiments.
- Immutable snapshots, persisted worker state, idempotent creation/start, branch/compare/cancel/export and standalone reproduction.
