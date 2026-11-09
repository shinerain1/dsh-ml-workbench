# Changelog

## 0.3.0

- Link experiments and studies to immutable DSH Session/tool-call identity; add bounded `ml_workbench_resume` recovery after restart or compaction.
- Deduplicate exact create/branch invocation replay automatically, retaining explicit request keys for new calls and PTC reruns.
- Declare typed canonical run/study/recovery fields for DSH PTC generation; keep Native policy, cancellation and plugin disposal behavior.
- Distinguish control timeouts from cancellation and explain how to recover possibly committed work.
- Fix a friction point observed in the live pilot: accept redundant native sklearn `random_state=seed` and `n_jobs=1`, while rejecting conflicting execution controls.
- Add controlled comparisons against direct MLflow and explicit-check positive controls, actual DSH Native-executor checks, and scripts for a live two-turn Agent pilot. No model-quality or general productivity gain is assumed.

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
