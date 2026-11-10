# Bounded workflows in DSH PTC

DSH owns TypeScript execution, tool discovery, guarded nested calls and cancellation. Enable its PTC executor/runtime in your DSH deployment; installing this plugin does not change the selected executor. The plugin's 20 tools work in Native mode too. Use stable intent keys across new outer calls. The returned canonical objects retain details that concise Native rendering omits.

Example program for DSH `run_code` (set the outer timeout to a budget supported by your DSH configuration). Replace the dataset path and target first:

```typescript
const base = {
  dataset_path: 'C:/project/customers.csv', target: 'repurchase',
  task: 'classification', seed: 42,
  test_fraction: 0.2, final_test_fraction: 0.2,
};
const created = await tools.ml_experiment_run({
  ...base, model: 'logistic_regression', request_key: 'customers-baseline-v1',
});
let run = created;
for (let i = 0; i < 2 && ['starting', 'running'].includes(run.state); i++) {
  run = await tools.ml_experiment_wait({run_id: run.run_id, wait_seconds: 30});
}
return {run_id: run.run_id, state: run.state, metrics: run.metrics};
```

An unfinished result is normal: return the ID and continue with another bounded call. Do not treat timeout of a control call as proof that creation failed. Resume/list/get before retrying. Use the same request key for the same intent and different keys for distinct experiments.

Compare successful candidates with `ml_experiment_compare`, which ranks validation scores only. Then call `ml_experiment_finalize` for the one selected model, followed by export. Finalization is never part of the automatic training/search loop. The final partition is committed to that model before any score is exposed. Repeating finalization for the same model is idempotent; another model using the same final partition is refused.

The actual provider test in `scripts/dsh-ptc-check.mjs` executes two fits, comparison, finalization and export in one outer program. It also replays the program, tries a forbidden filesystem write, and tests plugin shutdown. It uses real DSH Agents, PTC runtime and OS sandbox, with no LLM call or mocked PTC runtime. This measures execution plumbing and program composition, not an Agent's ability to author the program or general productivity.
