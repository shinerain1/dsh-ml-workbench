# DSH integration and recovery

Version 0.3.0 uses DSH's `defineTool` registry, immutable `ToolExecution` identity, Session identity, shared policy pipeline, cancellation signal and PTC schema generation. MLflow still owns tracking/models, Optuna owns study sampling/storage, and DSH owns reasoning and code generation. No agent-loop replacement, private prompt interception or second conversational memory is introduced.

## Session-aware operations

The adapter passes `exec.agent.session.id` and `exec.callId` in a separate RPC `dsh_context` envelope. These fields are not model arguments. Created runs record them in the local ledger, hashed manifest and MLflow tags `mlwb.dsh.session_id` / `mlwb.dsh.call_id`. Run creation commits its Session link and request key in the same ledger transaction. References to existing runs and studies are also linked after successful calls.

`ml_workbench_resume({})` returns bounded, paginated references for the calling Session. After compaction or restarting a Session, call it before planning new work. It returns each resource's `next_tool` and `arguments`; follow those to verify current metrics/artifacts or inspect a study. It neither starts training nor schedules a trial. A study link can represent an interrupted creation intent, so it is reported as unverified; retry the original `ml_search_create` if the study was not committed.

Use `previous_session_id` to navigate a different known Session, and `ml_experiment_list` for legacy/unlinked runs. This is a local convenience filter, not multi-user authorization. Keep the same backend `root` when restarting. The Session log already records normal tool calls/results; recovery information enters model context through a normal logged tool result, not invisible prompt injection.

## Retry identity

With no explicit `request_key`, create/branch derive a key from Session ID, call ID and operation. An exact invocation replay returns its committed run; changed arguments under the same identity fail with `IDEMPOTENCY_CONFLICT`. A new call ID is a new intent. Calls without an Agent retain the existing explicit-key behavior.

DSH PTC nested IDs are assigned by its executor. A rerun under a new outer call ID is a new identity. Always supply a stable `request_key` for intended retries across new model calls or PTC programs. Search creation already has the explicit immutable `study_name`. There is no semantic deduplication of similar natural-language instructions.

The local SQLite ledger and MLflow are separate stores. A crash before ledger commit can still leave a failed/incomplete MLflow run or files; the plugin does not claim a distributed exactly-once transaction. The tested lost-response case occurs after successful commit. Training start checks persisted worker identity and remains idempotent separately from creation.

## Native and PTC

Native mode exposes 17 tools. Output schemas declare run/study handles, states and resume references so DSH's generated PTC SDK can use canonical JSON fields without parsing rendered text. A PTC program can create a run with a stable key, start it, and return its ID in one model step. Poll in later bounded steps; do not keep one program open for the whole fit.

```typescript
const run = await tools.ml_experiment_create({
  dataset_path: "C:/project/customers.csv", target: "repurchase",
  task: "classification", model: "logistic_regression",
  request_key: "customer-baseline-v1"
});
const started = await tools.ml_experiment_start({ run_id: run.run_id });
return { run_id: started.run_id, state: started.state };
```

## Policy and lifetime boundaries

Calls enter DSH's ordinary execution pipeline. A `tools.guard()` denial or pre-aborted `exec.signal` prevents backend execution. Plugin unload aborts control requests and unregisters tools. Detached training is deliberately durable and survives the Session; use `ml_experiment_cancel` to stop it. It is not a DSH job owned by the disposed Agent, because that lifecycle would cancel the intended persistent fit.

Python still runs on the local host. Passing through DSH's tool policy does **not** place Python or custom factories inside DSH's shell/PTC OS sandbox. This adapter does not support remote SSH execution or hostile/multi-user workloads. Custom Python remains opt-in through `allowCustomCode`. Native/Python package versions must both be 0.3.0 or later for the new envelope and resume tool.

Sources: [DSH tool authoring](https://github.com/deepseek-ai/deepseek-harness/blob/8bd073392eef6f54e921a093b95076248eb08cbb/docs/cookbook/adding-a-tool.md), [execution identity](https://github.com/deepseek-ai/deepseek-harness/blob/8bd073392eef6f54e921a093b95076248eb08cbb/packages/core/tools/src/contracts.ts), [PTC executor](https://github.com/deepseek-ai/deepseek-harness/blob/8bd073392eef6f54e921a093b95076248eb08cbb/packages/core/tool-executor-ptc/README.md).
