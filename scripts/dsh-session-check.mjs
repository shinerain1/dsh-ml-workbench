/** Keyless behavior test with real DSH Agents and the production Native executor. */
import assert from 'node:assert/strict'
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import * as Plugin from '../lib/index.js'

if (!process.env.DSH_CHECKOUT) throw new Error('DSH_CHECKOUT must name a built DSH checkout')
const load = path => import(pathToFileURL(join(process.env.DSH_CHECKOUT, path)).href)
const { Context } = await load('vendor/cordis/lib/index.js')
const { mountAgentLoopTestDependencies, mountAgentLoopTestHarness } = await load('packages/test-support/agent-loop-testkit/lib/index.js')
await mkdir('.verification', { recursive: true })
const directory = await mkdtemp(resolve('.verification/dsh-session-'))
const root = join(directory, 'records')
const python = process.env.MLWB_PYTHON ?? resolve('.venv/Scripts/python.exe')
const dataset = join(directory, 'data.csv')
await writeFile(dataset, 'x,target\n' + Array.from({ length: 40 }, (_, i) => `${i},${i % 2}`).join('\n'))

async function mounted() {
  const ctx = new Context()
  await mountAgentLoopTestDependencies(ctx)
  const harness = await mountAgentLoopTestHarness(ctx)
  await ctx.plugin(Plugin, Plugin.Config({ root, pythonExecutable: python }))
  const a = await harness.create('mlwb-session-a')
  const b = await harness.create('mlwb-session-b')
  const call = async (agent, id, name, args = {}) => {
    const batch = await ctx.tools.execute({ agent, calls: [{ type: 'tool-call', id, name, arguments: JSON.stringify(args) }], signal: new AbortController().signal })
    return batch.results[0].result
  }
  return { ctx, a, b, call }
}

const args = { dataset_path: dataset, task: 'classification', target: 'target', model: 'logistic_regression' }
let h = await mounted()
const first = await h.call(h.a, 'create-1', 'ml_experiment_create', args)
assert.equal(first.isError, false, JSON.stringify(first))
assert.equal(first.value.dsh_context.session_id, 'mlwb-session-a')
await h.ctx.fiber.dispose()
// Recreate DSH objects with the persisted identity: no plugin memory survives.
// This tests plugin recovery, not DSH's own Session-log persistence implementation.
h = await mounted()
try {
  const replay = await h.call(h.a, 'create-1', 'ml_experiment_create', args)
  assert.equal(replay.isError, false, JSON.stringify(replay))
  assert.equal(replay.value.run_id, first.value.run_id)
  const another = await h.call(h.b, 'create-1', 'ml_experiment_create', args)
  assert.equal(another.isError, false, JSON.stringify(another))
  assert.notEqual(another.value.run_id, first.value.run_id)
  const resumed = await h.call(h.a, 'resume-1', 'ml_workbench_resume')
  assert.equal(resumed.isError, false, JSON.stringify(resumed))
  assert.deepEqual(resumed.value.resources.map(r => r.ref), [first.value.run_id])
  // Exercise the actual DSH policy seam: a deny prevents backend mutation.
  const dispose = h.ctx.tools.guard(exec => exec.name === 'ml_experiment_create' ? 'benchmark policy deny' : undefined)
  const denied = await h.call(h.a, 'denied-create', 'ml_experiment_create', args)
  assert.equal(denied.isError, true, JSON.stringify(denied))
  dispose()
  const after = await h.call(h.a, 'resume-2', 'ml_workbench_resume')
  assert.equal(after.value.resources.length, 1)
  const aborted = await h.ctx.tools.execute({ agent: h.a,
    calls: [{ type: 'tool-call', id: 'aborted', name: 'ml_experiment_create', arguments: JSON.stringify(args) }],
    signal: AbortSignal.abort(new Error('test cancellation')) })
  assert.equal(aborted.results[0].result.isError, true)
  const final = await h.call(h.a, 'resume-3', 'ml_workbench_resume')
  assert.equal(final.value.resources.length, 1)
  const report = { kind: 'keyless-native-executor', real_dsh_agents: true, model_calls: 0,
    same_call_replay_unique_runs: 1, other_session_unique_runs: 1, resume_resources: 1,
    policy_denial_prevented_creation: true, pre_aborted_call_prevented_creation: true,
    plugin_restart_recovered: true, dsh_session_log_reload_tested: false }
  await writeFile('docs/dsh-session-results.json', JSON.stringify(report, null, 2) + '\n')
  console.log(JSON.stringify(report))
} finally { await h.ctx.fiber.dispose() }
