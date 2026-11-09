# Optuna search through DSH

The plugin uses Optuna's **actual TPE sampler, ask/tell API and RDB storage**. MLflow keeps every fitted experiment. Optuna stores trials and suggested parameters in `optuna.sqlite3`; each trial links to its MLflow run. This is a sequential, agent-driven scheduler: one `ml_search_step` call reconciles current work and starts at most one trial. Training runs independently of the control call.

Create or choose a v0.2 experiment using a built-in estimator, then call:

```json
{"study_name":"forest-depth-01","run_id":"<base run ID>","n_trials":12,"metric":"roc_auc","search_space":{"max_depth":{"type":"int","low":2,"high":12},"min_samples_leaf":{"type":"int","low":1,"high":6}},"reuse_preprocessing":false,"trial_timeout_seconds":600}
```

`ml_search_create` does not fit. `study_name` is an immutable idempotency identifier (letters, digits, dots, underscores and hyphens, max 80 characters). Repeating an identical create returns the same study; changing its configuration under the same name is refused. Use a new study for a new budget or search space.

Then repeatedly call `ml_search_step` with `{"study_name":"forest-depth-01"}` until `state` is `complete` or `cancelled`. It returns trial states, linked run IDs and the verified best completed run. Use `ml_experiment_export` or `ml_model_predict` with that run ID. `ml_search_get` reconciles results without scheduling another trial. `ml_search_cancel` persists cancellation and stops the active fit, confirming process exit before reporting completion.

Search distributions:

```json
{
  "C": {"type":"float","low":0.001,"high":10.0,"log":true},
  "solver": {"type":"categorical","choices":["lbfgs","liblinear"]}
}
```

Only estimator parameters can be tuned; `seed` is fixed for the underlying experiment and `n_jobs=1`. Parameters not listed in the search space retain the base run's values. Bounds and parameter names are validated. Unsupported parameter values fail their trial and count against the finite 1–100 trial budget. A custom preprocessing factory may be used when enabled, but custom estimator search is not supported.

Default `reuse_preprocessing=false` refits preprocessing on the same training rows for each trial. Setting it true requires a successful base and reuses its fitted transformer. Data, row split and evaluator must remain identical. Group/time splits are inherited from the base experiment.

Crash handling:

- The branch request is persisted before creating a child MLflow run, with a stable study/trial request key.
- If creation committed but its response or run link was lost, the controller recovers that child by its request key.
- If the fit was started but the response was lost, repeated `start` observes its durable run state and does not launch another worker.
- A crash while sampling before any prepared request marks that abandoned Optuna trial FAIL. It does not invent a completed experiment.
- DSH can exit and reopen; calling step with the same study name resumes scheduling. There is no separate background study coordinator.

The TPE sampler is reseeded for each new trial with `(seed + trial_count) mod 2^32`, using the persisted trial history. This makes controller restarts consistent with this adapter's scheduling policy; it is not a claim of bit-for-bit parity with one continuously running `study.optimize` call.

**Timeout semantics:** elapsed fit time is checked when step/get is polled. When DSH is closed, the existing fit continues and no new trial starts. This is not a hard background time limit. Use cancel to stop running work explicitly.

The best value is a validation selection result, **not an independent test score**. Repeatedly tuning against the same holdout increases selection bias. Keep a separate final dataset appropriate to the actual customer/time boundary; the plugin does not manufacture that scientific guarantee.
