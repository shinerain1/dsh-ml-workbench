# DSH ML Workbench 0.3.0

Experiments now follow DSH Session and tool-call identity. `ml_workbench_resume` recovers run/study references; exact create/branch replays deduplicate automatically. Canonical result schemas expose typed PTC handles. Timeout messages explain recovery when a response is lost.

A live DSH pilot exposed rejection of redundant sklearn `random_state` and `n_jobs` values. Matching values are now normalized; conflicting controls remain errors. Both original failed payloads pass without duplicating runs.

Validation includes 31 distinct Python cases, four Node cases, Native-executor checks, four direct-MLflow comparisons and one live two-turn trial per arm. The pilot showed identical accuracies and successful continuation in both arms, but more tool calls with the plugin. The plugin delivery retrained its estimator; the direct workflow reloaded and rescored. The report explains the prompt's ambiguity and does not present this as an established productivity win.

Install **both** the 0.3.0 Python wheel and DSH bundle. Existing runs remain usable; old unlinked runs are accessible through `ml_experiment_list`. This is a local trusted-user adapter, not a full upstream-platform replacement or sandboxed training service.
