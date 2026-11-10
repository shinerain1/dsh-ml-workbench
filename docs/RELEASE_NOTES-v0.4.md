# v0.4.0

New experiments reserve train/validation/final-test partitions. Training and Optuna see validation scores only; finalize commits one selected model per final partition before exposing final-test scores. Export and standalone retraining verify both metric sets. Existing two-way snapshots retain their original partitions and frozen recipes.

DSH now has 20 tools, including combined run, bounded wait and finalize. PTC retains complete canonical objects while Native rendering omits repetitive histories and detailed classification tables. Real DSH PTC, OS confinement and process-range shutdown were exercised; the implementation uses DSH services, not a replacement Agent executor.

Every new fit has a supervisor with a persisted wall-clock deadline. Trial timeouts work without polling. Local execution remains durable across DSH shutdown; the optional DSH execution mode applies current DSH file policy and stops owned workers on plugin/DSH shutdown. Missing confinement services fail closed. Windows reports partial file confinement, with no network/read-isolation claim.

An installation helper creates/reuses an isolated environment, installs locked dependencies, checks backend/plugin version equality, diagnoses dependencies/SQLite/Node, and generates a configuration snippet without overwriting the user's DSH profile.

Limits: local single-user CSV workflows; no hostile-code boundary, remote SSH backend, unattended search scheduler, nested CV, checkpoint resumption or general productivity claim. The earlier live Agent pilot remains a one-pair exploratory result. See [validation evidence](VALIDATION-v0.4.md) and [maturity review](MATURITY.md).
