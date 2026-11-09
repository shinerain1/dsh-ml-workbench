# Does the adapter improve the workflow?

The measured benefit is automatic enforcement and DSH integration. It does not improve the underlying learning algorithm. A carefully written MLflow workflow can implement the same safeguards. We tested that positive control instead of assuming that direct MLflow is incapable.

## Controlled mechanism experiment

Run `scripts/benchmark-workflow.py` with a fresh `--root` and a `--report` path. [Raw results](benchmark-results.json) contain all four paired runs: classification with split seeds 7, 19 and 41, and regression with seed 7. Each arm receives identical synthetic data, split, model parameters and frozen sklearn recipe. The direct arm logs the full inputs/model using native `MlflowClient`; it does not call Workbench during execution. Sharing the recipe deliberately isolates workflow behavior and gives the baseline reproducible code.

| Observed result | Direct MLflow | Plugin |
|---|---:|---:|
| Paired model metrics | Identical in all 4 pairs | Maximum absolute difference 0 |
| Required record categories present | 8/8 in each run | 8/8 in each run |
| Reproductions matching recorded metrics | 4/4 | 4/4 standalone ZIP reproductions |

The direct reproductions use their logged inputs in the development interpreter. Plugin ZIPs are reproduced by the existing isolated release interpreter with the same locked numerical dependencies. This establishes reproducibility in these tested environments, not portability to arbitrary environments. Eight record categories are data, config, split, recipe, environment, requirements, model and metrics.

| Injected fault | Direct MLflow primitive | Direct MLflow with explicit check | Plugin |
|---|---|---|---|
| Successful creation response lost, then replayed | Creates 2 runs | Request tag lookup returns 1 | Same DSH call returns 1 |
| Ranking scores from different validation rows | Returns a ranking | Split-hash check rejects | Compare rejects |
| Model file changed after success | `get_run` still reports FINISHED | Saved hash check rejects | Export rejects |

These are deliberately selected failure cases, not measured incident rates. The explicit tag lookup is a serial positive control, not a concurrency-safe idempotency service. The plugin's lost-response guarantee covers a committed call; it is not a distributed transaction across MLflow, SQLite and the filesystem.

## Native DSH integration experiment

`scripts/dsh-session-check.mjs` uses real DSH Agent objects and its production Native tool executor at checkout `8bd073392eef6f54e921a093b95076248eb08cbb`. [Results](dsh-session-results.json) confirm exact replay returns one run, another Session remains separate, recreating the plugin with the same Session identity recovers references, and a DSH guard denial or pre-aborted signal prevents creation. No LLM is involved in this test. Reconstructing Agent objects is not itself a test of DSH's on-disk Session log reader.

## Live Agent pilot protocol

`prepare-agent-pilot.py` and `run-agent-pilot.py` run the actual source DSH headless profile with `deepseek-official/deepseek-flash`, low reasoning effort, output cap 8192 and no provider retries. Each arm gets the same 160-row synthetic CSV and two turns. The first trains RandomForest and ExtraTrees on a common stratified split. The second launches a fresh DSH process adopting the persisted Session, reuses the second fit's exact preprocessing for logistic regression, and exports/reproduces the new model. The plugin arm additionally exposes the 17 tools; the other can use installed MLflow/sklearn normally. Neither imports the adapter in Python or installs packages.

Use an independently writable Windows directory. Initial attempts inside the Codex workspace encountered `SetNamedSecurityInfoW (Win32 5)` from DSH's sandbox and are excluded from task-success comparisons. DSH's sandbox is retained in the eligible pilot. All raw logs remain local because they include machine paths and model reasoning. A small pilot cannot establish a general reduction in human intervention, token cost or failure probability. Wall-clock timings are especially confounded by cold imports, provider latency and concurrent verification processes.

## Live pilot observations

[Audited results](agent-pilot-results.json) record one completed two-turn run per arm. Both restored the same persisted DSH Session in a fresh process, preserved the original models, copied the parent's fitted preprocessor byte-for-byte and delivered a package whose metrics were independently reproduced in an isolated interpreter. The accuracies were **0.8125, 0.84375 and 0.84375 in both arms**. Both used the same partition membership; the baseline sorted row indices and the plugin retained splitter order. There is no model-quality advantage demonstrated.

| Observation across both turns | Without plugin | With plugin, before the parameter fix |
|---|---:|---:|
| Extra user answers | 0 | 0 |
| Tool calls | 22 | 36 |
| Executor-reported tool errors | 0 | 2 |
| Provider output tokens | 27,314 | 9,462 |
| Provider total tokens including cache reads | 592,196 | 704,042 |

The plugin generated less output in this single pilot but made more calls and consumed more total tokens. These are observations, not statistically supported efficiency or billing claims. Both arms also used ordinary DSH filesystem/PowerShell tools to verify artifacts.

The two errors were redundant sklearn controls: the model supplied `params.random_state=42` and `params.n_jobs=1` alongside `seed=42`. The previous validator rejected even matching values. Version 0.3 now normalizes matching integers while rejecting conflicts, nulls and booleans. [Exact-payload regression results](pilot-regression-results.json) show both failed calls now succeed and return the existing run IDs, with three total experiments remaining. This is a deterministic replay, **not a rerun of the live pilot**; the original 36 calls and two errors are retained above.

An independent post-hoc check instrumented `LogisticRegression.fit` during standalone reproduction. The baseline called it **0 times** (reload and rescore); the plugin called it **1 time** (train a new estimator using copied preprocessing and verify metrics). Both reproduced scores, but only the plugin delivery demonstrated retraining. The initial prompt did not explicitly require refitting, so the baseline is not counted as failing a preregistered criterion. Future repetitions can use `prepare-agent-pilot.py --strict-retrain` to make that requirement explicit for both arms. `audit-agent-pilot.py` separates those outcomes mechanically.

The evidence supports automatic safeguards, Session linkage and an explicit delivery procedure. It does **not** establish fewer human interventions, fewer calls, lower cost, better model accuracy or maturity equivalent to upstream platforms.
