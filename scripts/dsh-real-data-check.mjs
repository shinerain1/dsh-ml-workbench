/** Real DSH PTC/session/sandbox integration on UCI retail; zero LLM calls. */
import assert from 'node:assert/strict'
import { copyFile, mkdir, mkdtemp, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import * as Plugin from '../lib/index.js'

if (!process.env.DSH_CHECKOUT || !process.env.MLWB_RETAIL_CSV) throw new Error('Set DSH_CHECKOUT and MLWB_RETAIL_CSV')
const load = path => import(pathToFileURL(join(process.env.DSH_CHECKOUT, path)).href)
const { Context } = await load('vendor/cordis/lib/index.js')
const { mountAgentLoopTestDependencies, mountAgentLoopTestHarness } = await load('packages/test-support/agent-loop-testkit/lib/index.js')
const directory = await mkdtemp(join(tmpdir(), 'dsh-mlwb-real-data-'))
const dataset = join(directory, 'retail.csv')
await copyFile(process.env.MLWB_RETAIL_CSV, dataset)
const ctx = new Context()
let nested = 0
let report
try {
  for (const name of ['subprocess', 'sandbox']) {
    const module = await load(`packages/${name}/${name}-local/lib/index.js`)
    await ctx.plugin(module.default, {})
  }
  const Policy = (await load('packages/sandbox/sandbox-policy/lib/index.js')).default
  const Runtime = (await load('packages/ptc-runtime/ptc-runtime-node/lib/index.js')).default
  const policy = ctx.plugin(Policy, { mode: 'workspace-write', workspaceRoot: directory })
  const runtime = ctx.plugin(Runtime, { timeoutMs: 600000, maxTimeoutMs: 600000 })
  await mountAgentLoopTestDependencies(ctx, { workingDirectory: true, tools: { mode: 'ptc' } })
  await policy
  await runtime
  const harness = await mountAgentLoopTestHarness(ctx)
  const config = Plugin.Config({ root: join(directory, 'records'), executionMode: 'dsh',
    pythonExecutable: process.env.MLWB_PYTHON ?? resolve('.venv/Scripts/python.exe') })
  let plugin = ctx.plugin(Plugin, config)
  await plugin
  ctx.tools.guard(exec => { if (exec.name.startsWith('ml_')) nested++; return undefined })
  const agent = await harness.create('mlwb-real-data', {}, { cwd: directory })
  const call = async (code, id) => {
    const batch = await ctx.tools.execute({ agent, calls: [{ type: 'tool-call', id, name: 'run_code',
      arguments: JSON.stringify({ code, description: 'Run the retail experiment workflow', timeoutMs: 600000 }) }], signal: AbortSignal.timeout(650000) })
    const result = batch.results[0].result
    assert.equal(result.isError, false, JSON.stringify(result))
    return result.value.result
  }
  const code = `
    async function finished(run) {
      for(let i=0;i<12&&['starting','running'].includes(run.state);i++)
        run=await tools.ml_experiment_wait({run_id:run.run_id,wait_seconds:30});
      if(run.state!=='succeeded') throw new Error(JSON.stringify(run));
      return run;
    }
    const parent=await finished(await tools.ml_experiment_run({dataset_path:${JSON.stringify(dataset)},
      target:'target',task:'classification',model:'logistic_regression',params:{max_iter:1000},seed:7,
      split:{strategy:'time',column:'anchor'},preprocessing:{drop_columns:['customer_id']},
      test_fraction:0.2,final_test_fraction:0.2,request_key:'retail-parent'}));
    const branch=await tools.ml_experiment_branch({run_id:parent.run_id,
      changes:{model:'hist_gradient_boosting',params:{max_iter:80,max_leaf_nodes:15}},
      reuse_preprocessing:true,request_key:'retail-branch'});
    const child=await finished(await tools.ml_experiment_start({run_id:branch.run_id}));
    const comparison=await tools.ml_experiment_compare({run_ids:[parent.run_id,child.run_id],metric:'average_precision'});
    const selected=comparison.ranking[0].run_id;
    const final=await tools.ml_experiment_finalize({run_id:selected});
    const exported=await tools.ml_experiment_export({run_id:selected,include_data:true});
    return {run_ids:[parent.run_id,child.run_id],selected,reused_preprocessing:child.reused_preprocessing,
      validation:comparison.ranking.map(r=>({model:r.model,metrics:r.metrics})),final_test:final.final_test.metrics,
      delivery:exported.path};`
  const first = await call(code, 'real-workflow')
  const normalCalls = nested
  console.log(JSON.stringify({ stage: 'workflow', nested_calls: normalCalls, selected: first.selected }))
  const replay = await call(code, 'real-workflow-replay')
  assert.deepEqual(replay.run_ids, first.run_ids)
  assert.deepEqual(replay.final_test, first.final_test)
  assert.equal(first.reused_preprocessing, true)
  await plugin.dispose()
  plugin = ctx.plugin(Plugin, config)
  await plugin
  const resumed = await call('return await tools.ml_workbench_resume({});', 'real-resume')
  const found = resumed.resources.map(r => r.ref)
  assert(first.run_ids.every(id => found.includes(id)), JSON.stringify(resumed))
  report = { kind: 'real-data-real-dsh-ptc', model_calls: 0, execution_mode: 'dsh',
    policy: 'workspace-write', platform: process.platform, outer_calls_for_workflow: 1,
    nested_calls_for_workflow: normalCalls, replay_same_run_ids: true, replay_same_final_metrics: true,
    plugin_remount_session_recovered: true, completed: first,
    limitations: ['No LLM calls or Agent token/human-intervention comparison.',
      'Plugin remount reuses the DSH Session; this is not a full DSH process restart.',
      'This separate integration workspace repeats a fixed workflow; it is not another independent statistical test.'] }
} finally {
  await ctx.fiber.dispose()
}
report.context_disposed = true
await mkdir('.verification', { recursive: true })
await writeFile('.verification/dsh-real-data-local.json', JSON.stringify(report, null, 2) + '\n')
const publicReport = JSON.parse(JSON.stringify(report).replaceAll(JSON.stringify(directory).slice(1, -1), '<temporary-workspace>'))
await writeFile('docs/dsh-real-data-results.json', JSON.stringify(publicReport, null, 2) + '\n')
console.log(JSON.stringify({ ok: true, report: 'docs/dsh-real-data-results.json' }))
