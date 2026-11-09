# Validation evidence

This report describes executed checks, not a proposed experiment or a claim of improved agent productivity. The sample data is generated using scikit-learn; no customer records are used or published.

## Executed checks

- Python: **9 tests passed** (581.06 s) on Windows / Python 3.12.0. They perform real classification/regression fits and verify saved-model predictions, MLflow artifact linkage, exact fitted-preprocessor reuse, independent reproduction, incompatible comparisons, idempotency, input/metric tampering, concurrent starts, confirmed cancellation, worker death, restart as a branch, caller exit and PID-reuse safety. One upstream SQLAlchemy deprecation warning was emitted; it did not fail a test.
- Node: **3 tests passed**. The compiled plugin mounts in the official Cordis/DSH Tools services, executes create/start/get/export through the tool executor and removes its nine tools on disposal. Transport cancellation and executable failure are checked.
- Actual DSH source checkout: the plugin mounts, exposes nine schemas, executes health through DSH's native batch tool executor and unloads successfully. Tested checkout: `8bd073392eef6f54e921a093b95076248eb08cbb`, package version `0.2.2-alpha.3`.
- TypeScript build succeeded. Python lint checks and packaging verification are recorded in the release evidence.
- Four additional protocol cases passed (0.17 s), rejecting malformed creation idempotency keys before side effects. Total Python cases: **13**.
- The `.tgz` was installed by the actual DSH CLI plugin manager into an isolated headless profile. The resolved profile includes the installed bundle's `ml-workbench` entry. pnpm reported peer-dependency warnings; native-host execution is separately tested above.
- The **installed** bundle was then executed against the actual DSH native tool executor using a fresh virtual environment containing the release wheel. Health and unload passed, and `pip check` found no broken dependencies.
- [packaging-results.json](packaging-results.json) records independent reconstruction of both exported deliveries in the new virtual environment. Both sets of reproduced metrics match; the imported backend came from the installed wheel in site-packages, not the editable source tree.

An online headless DSH request was attempted using the configured account provider. It stopped before inference with `ACCOUNT_SIGN_IN_REQUIRED`. No online natural-language workflow or native-versus-plugin agent trial is claimed for this release. Existing account credentials were neither printed nor included in artifacts. The user can complete this check after signing in to DeepSeek in DSH.

## Recomputed synthetic workflow

[validation-results.json](validation-results.json) contains exact machine-readable results from `scripts/demo.py`. The script executes seven successful experiments, including a deliberately incompatible split, and independently retrains two exported deliveries. Both deliveries reproduced their recorded metrics.

| Synthetic task | Model | Result |
|---|---|---|
| Classification, 600 rows / fixed 20% holdout | Random forest, 60 trees | Accuracy 0.891667 |
| Same classification split | Extra trees, 60 trees, reused preprocessing | Accuracy 0.891667 |
| Same classification split | Third experiment: custom robust preprocessing + forest | Accuracy 0.891667 |
| Same classification split | Fourth experiment: third's fitted preprocessing + logistic regression | Accuracy 0.700000 |
| Regression, 400 rows / fixed 20% holdout | Ridge | RMSE 5.570744 |
| Same regression split | Random forest, 40 trees | RMSE 47.163306 |

The linear classifier was faster in the single local measurement but less accurate. This is an observed trade-off on this generated dataset, not a universal speedup. Timing varies across machines and is not a multi-seed benchmark. The different-split comparison was refused with `INCOMPARABLE_EXPERIMENTS`.

Re-run in a new directory:

```powershell
.\.venv\Scripts\python.exe scripts/demo.py --root .verification/demo-2 --report docs/validation-results.json
```

## Agent-comparison protocol and current limits

The deterministic tests above establish workflow behavior. They do **not** measure native DSH versus plugin-assisted DSH, human intervention rate or real business performance. The JSON report leaves human intervention rate null instead of inventing a result.

A future controlled agent study should pin DSH/model/version and prompts, use the same CSV and compute budget, and run both conditions repeatedly from isolated workspaces. Give each condition: initial three-model training; an interruption; a follow-up preserving experiment three's preprocessing while changing the model; a request to compare incompatible splits; and export/reproduction in a clean environment. Keep the model, data, randomization and intervention policy identical across conditions.

Record human interventions under a predeclared rubric; complete experiment records with code/data/parameter/model linkage divided by all attempted experiments; and independently reproduced deliveries divided by all attempted deliveries. Count failures, timeouts and unrecoverable runs in denominators. Report raw logs with sensitive values removed, repeated-trial uncertainty and whether the result generalizes beyond these tasks. Do not infer such rates from unit tests.
