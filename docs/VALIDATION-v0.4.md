# v0.4 validation

Windows, Python 3.12.0, Node 24.18.0. DSH source commit `8bd073392eef6f54e921a093b95076248eb08cbb` (`0.2.2-alpha.3`). Core libraries and exact versions are recorded in [test evidence](test-results-v4.json) and requirements.lock. This validates a local single-user adapter, not parity with complete upstream platforms.

## Executed checks

| Check | Evidence and scope |
|---|---|
| Python workflows and reliability | 36 distinct tests, all passing after fixes; [case list](test-results-v4.json) |
| Node transport, registry and fail-closed confinement | 6 tests passed: actual registry lifecycle, pre-abort, missing interpreter, timeout cleanup, missing DSH services, confinement failure without spawn/permission expansion |
| Actual DSH PTC and OS sandbox | [provider results](dsh-ptc-v4-results.json): two fits, validation ranking, final-test evaluation and export in one outer program, 7 nested tools; new outer-call replay retained the same run IDs and final scores |
| Forbidden filesystem effect | A reviewed test factory attempted an outside-workspace write; the actual Windows provider denied it with PermissionError; no marker file was created |
| DSH worker lifetime | Disposing the plugin stopped a slow fit and its training child; a subsequent backend query marked the experiment interrupted |
| Background deadlines | A deliberately sleeping factory exceeded its deadline and was killed; the test waited on process exit without get/search polling and checked the durable FAILED state and child exit |
| Supervisor crash | Killing the supervisor caused its training child to exit; restart correctly reported interrupted |
| Legacy migration | [actual v0.3 source test](upgrade-v4-results.json) trained a legacy run, then v0.4 reused its frozen preprocessor/recipe, compared and exported it |
| Final evaluation and delivery | The selected branch finalized idempotently; another candidate sharing its final partition was refused. Standalone reproduction refitted a new estimator and matched validation and final-test metrics. Mutated final artifacts blocked export; interrupted final-artifact commits also verify every recomputed file before recovery |

The full Python pass contained 34 passes and one failure from Python bytecode being written into an immutable snapshot during finalization. After changing recipe loading to avoid that write, the affected end-to-end case passed (including refit/export). A separate Windows SQLite diagnostic cleanup test passed after explicitly closing its connection. The reported 36 cases deduplicate the rerun; no failed case is counted as passing without a successful rerun. Earlier supervisor tests also found and fixed Windows venv launcher PID handoff. Development failure logs remain in local .verification; published results list the final outcomes.

Packaging evidence is recorded separately in [installed-wheel and reproduction results](packaging-v4-results.json). A new isolated venv was created with the installation helper. The final installation diagnostic passed; the helper generated a config snippet without touching a DSH profile. Separate classification/regression delivery copies refitted new estimators and matched validation/final-test scores. The verification harness explicitly decodes UTF-8 JSON so Chinese workspace paths work consistently. The installed wheel also passed actual DSH Native registry/health/unload with 20 tools.

## What this establishes

The new mechanisms prevent specific observed workflow mistakes: automatic tuning never evaluates the final partition, a second candidate cannot claim an already-selected final partition, polling is unnecessary for training timeout, and DSH mode cannot silently fall back to unconfined host spawning. The actual PTC program demonstrates that seven nested operations can be composed without seven model responses. It does not measure an LLM's ability to write that program or its token/time cost.

The [earlier live Agent pilot](BENCHMARK.md) remains exploratory and unfavorable on tool count (36 versus 22 before the reported parameter fix). There is no new randomized live-model A/B trial in this release, and no claim of improved prediction accuracy, lower total cost or generalized reduction in human intervention. The PTC fixture is synthetic and tests plumbing, not business usefulness of its scores.

## Boundaries

- Default fractions are 60/20/20; group/time constraints and small datasets can change exact row counts or cause rejection. Labels and business leakage still need human judgment.
- Final selection is a local workflow ledger, not access control over the original CSV, a signed audit trail, or protection against intentionally using another root/partition. Frozen final metrics do not make repeatedly reused test data scientifically independent.
- Windows DSH reports partial filesystem confinement. Reads and networking are not restricted by this plugin. Custom code and serialized models must remain trusted. Remote providers, Linux and macOS have not been validated here.
- Local mode survives DSH shutdown; DSH mode uses managed processes and stops on plugin/DSH shutdown. Policy is resolved at launch; cancel existing fits before changing their intended permissions. There is no model-weight checkpoint resume.
- A deadline is enforced by a local supervisor and normal OS process termination, not a real-time operating system guarantee. Backend initialization, filesystem stalls and machine suspension can delay status updates. No study scheduler starts the next trial without DSH step calls.
- Every control call still starts Python; this has measurable overhead. No speed claim is made from cold-start timings.
- MLflow emitted a SQLAlchemy deprecation warning; the tested workflows passed. Locked dependencies are provided for reproducibility, not a claim that the adapter is production-certified.
