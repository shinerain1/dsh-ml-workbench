/** Actual DSH PTC + Windows/Linux/macOS sandbox providers, without a model or fake runtime. */
import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { access, mkdir, mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import * as Plugin from '../lib/index.js'

if (!process.env.DSH_CHECKOUT) throw new Error('DSH_CHECKOUT must name a built DSH checkout')
const load = path => import(pathToFileURL(join(process.env.DSH_CHECKOUT, path)).href)
const { Context } = await load('vendor/cordis/lib/index.js')
const { mountAgentLoopTestDependencies, mountAgentLoopTestHarness } = await load('packages/test-support/agent-loop-testkit/lib/index.js')
const directory = await mkdtemp(join(tmpdir(), 'dsh-mlwb-v4-ptc-'))
const python = process.env.MLWB_PYTHON ?? resolve('.venv/Scripts/python.exe')
const root = join(directory, 'records')
const dataset = join(directory, 'data.csv')
await writeFile(dataset, 'x,z,target\n' + Array.from({ length: 120 }, (_, i) => `${i % 17},${i % 11},${i % 4 > 1 ? 1 : 0}`).join('\n'))
const outside = resolve('.verification', 'denied-' + randomUUID() + '.txt')
await mkdir(resolve('.verification'), { recursive: true })
const factory = join(directory, 'denied.py')
await writeFile(factory, `from pathlib import Path\ndef make_estimator(task, params, seed):\n    Path(${JSON.stringify(outside)}).write_text('must be denied', encoding='utf-8')\n    raise RuntimeError('sandbox failed to deny outside write')\n`)
const ctx = new Context()
let nested = 0
try {
  for (const name of ['subprocess', 'sandbox']) {
    const module = await load(`packages/${name}/${name}-local/lib/index.js`)
    await ctx.plugin(module.default, {})
  }
  const Policy = (await load('packages/sandbox/sandbox-policy/lib/index.js')).default
  const Runtime = (await load('packages/ptc-runtime/ptc-runtime-node/lib/index.js')).default
  const policyFiber = ctx.plugin(Policy, { mode: 'workspace-write', workspaceRoot: directory })
  const runtimeFiber = ctx.plugin(Runtime, { timeoutMs: 600000, maxTimeoutMs: 600000 })
  await mountAgentLoopTestDependencies(ctx, { workingDirectory: true, tools: { mode: 'ptc' } })
  await policyFiber
  await runtimeFiber
  const harness = await mountAgentLoopTestHarness(ctx)
  const plugin = ctx.plugin(Plugin, Plugin.Config({ root, pythonExecutable: python, executionMode: 'dsh', allowCustomCode: true }))
  await plugin
  ctx.tools.guard(exec => { if (exec.name.startsWith('ml_')) nested++; return undefined })
  const agent = await harness.create('mlwb-v4-ptc', {}, { cwd: directory })
  const call = async (code, id) => {
    const batch = await ctx.tools.execute({ agent, calls: [{ type: 'tool-call', id, name: 'run_code',
      arguments: JSON.stringify({ code, description: 'Validate the ML experiment workflow', timeoutMs: 600000 }) }], signal: AbortSignal.timeout(650000) })
    const result = batch.results[0].result
    assert.equal(result.isError, false, JSON.stringify(result))
    return result.value
  }
  const code = `
    const base = { dataset_path: ${JSON.stringify(dataset)}, target: 'target', task: 'classification', final_test_fraction: 0.2 };
    async function finished(run) {
      for (let i = 0; i < 5 && ['starting','running'].includes(run.state); i++)
        run = await tools.ml_experiment_wait({run_id:run.run_id, wait_seconds:30});
      if (run.state !== 'succeeded') throw new Error(JSON.stringify(run));
      return run;
    }
    const a = await finished(await tools.ml_experiment_run({...base, model:'logistic_regression', request_key:'ptc-a'}));
    const b = await finished(await tools.ml_experiment_run({...base, model:'extra_trees', params:{n_estimators:8}, request_key:'ptc-b'}));
    const comparison = await tools.ml_experiment_compare({run_ids:[a.run_id,b.run_id]});
    const selected = comparison.ranking[0].run_id;
    const final = await tools.ml_experiment_finalize({run_id:selected});
    const delivery = await tools.ml_experiment_export({run_id:selected,include_data:true});
    return {run_ids:[a.run_id,b.run_id], selected, validation:comparison.ranking.map(r=>r.metrics.accuracy), final_test:final.final_test.metrics, delivery:delivery.path};`
  const completed = await call(code, 'workflow')
  const normalCalls = nested
  const replayed = await call(code, 'workflow-replay')
  assert.deepEqual(replayed.result.run_ids, completed.result.run_ids)
  assert.deepEqual(replayed.result.final_test, completed.result.final_test)
  const denied = await call(`const run = await tools.ml_experiment_run({dataset_path:${JSON.stringify(dataset)},target:'target',task:'classification',model:'custom',estimator_path:${JSON.stringify(factory)},request_key:'denied'}); let state=run; for(let i=0;i<4&&['starting','running'].includes(state.state);i++) state=await tools.ml_experiment_wait({run_id:run.run_id,wait_seconds:30}); if(state.state!=='failed'||!String(state.error).includes('PermissionError')) throw new Error(JSON.stringify(state)); return {state:state.state,denied:true};`, 'denial')
  await assert.rejects(access(outside))
  const slow = join(directory, 'slow.py')
  await writeFile(slow, 'import time\nfrom sklearn.linear_model import LogisticRegression\ndef make_estimator(task, params, seed):\n    time.sleep(120)\n    return LogisticRegression()\n')
  const active = await call(`const r = await tools.ml_experiment_run({dataset_path:${JSON.stringify(dataset)},target:'target',task:'classification',model:'custom',estimator_path:${JSON.stringify(slow)},request_key:'shutdown'}); return await tools.ml_experiment_wait({run_id:r.run_id,wait_seconds:15});`, 'shutdown')
  assert.equal(active.result.state, 'running')
  await plugin.dispose()
  const read = await promisify(execFile)(python, ['-c', 'import json,sys; from pathlib import Path; from dsh_ml_workbench.service import Workbench,owned_process; w=Workbench(Path(sys.argv[1])); r=w.get(sys.argv[2]); child={"pid":r.get("training_pid"),"process_created_at":r.get("training_process_created_at")}; print(json.dumps({"state":r["state"],"child_alive":owned_process(child) is not None}))', root, active.result.run_id], { windowsHide: true, env: { ...process.env, PYTHONUTF8: '1' }, timeout: 60000 })
  const shutdown = JSON.parse(read.stdout)
  assert.deepEqual(shutdown, { state: 'interrupted', child_alive: false })
  const report = { kind: 'real-dsh-ptc-sandbox', model_calls: 0, outer_calls_for_workflow: 1,
    nested_calls_for_workflow: normalCalls, execution_mode: 'dsh', workspace_policy: 'workspace-write',
    platform: process.platform, completed, replayed, outside_write_denied: true, denied, shutdown }
  await writeFile('.verification/dsh-ptc-v4-local.json', JSON.stringify(report, null, 2) + '\n')
  const publicReport = JSON.parse(JSON.stringify(report).replaceAll(JSON.stringify(directory).slice(1,-1), '<temporary-workspace>'))
  await writeFile('docs/dsh-ptc-v4-results.json', JSON.stringify(publicReport, null, 2) + '\n')
  console.log(JSON.stringify(publicReport))
} finally { await ctx.fiber.dispose() }
