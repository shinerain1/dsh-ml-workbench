/** Re-execute the two observed failed model calls after normalization was fixed. */
import assert from 'node:assert/strict'
import { readFile, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import * as Plugin from '../lib/index.js'

const root = resolve(process.argv[2])
const load = path => import(pathToFileURL(join(process.env.DSH_CHECKOUT, path)).href)
const { Context } = await load('vendor/cordis/lib/index.js')
const { mountAgentLoopTestDependencies, mountAgentLoopTestHarness } = await load('packages/test-support/agent-loop-testkit/lib/index.js')
const events = (await readFile(join(root, 'turn-1.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse)
const failures = events.filter(e => e.type === 'tool_result' && e.status === 'error')
assert.equal(failures.length, 2)
const expected = JSON.parse(await readFile(join(root, 'pilot-result.json'), 'utf8')).experiments.slice(0, 2).map(e => e.run_id)
const ctx = new Context()
try {
  await mountAgentLoopTestDependencies(ctx)
  const harness = await mountAgentLoopTestHarness(ctx)
  const agent = await harness.create('control-normalization-regression')
  await ctx.plugin(Plugin, Plugin.Config({ root: join(root, 'records'), pythonExecutable: process.env.MLWB_PYTHON ?? resolve('.venv/Scripts/python.exe') }))
  for (const failure of failures) {
    const call = events.find(e => e.type === 'tool_call' && e.callId === failure.callId)
    assert.equal(call.tool, 'ml_experiment_create')
    const result = await ctx.tools.execute({ agent, calls: [{ type: 'tool-call', id: call.callId,
      name: call.tool, arguments: JSON.stringify(call.input) }], signal: new AbortController().signal })
    assert.equal(result.results[0].result.isError, false, JSON.stringify(result))
    assert.ok(expected.includes(result.results[0].result.value.run_id))
  }
  const listed = await ctx.tools.execute({ agent, calls: [{ type: 'tool-call', id: 'list', name: 'ml_experiment_list', arguments: '{}' }], signal: new AbortController().signal })
  assert.equal(listed.results[0].result.value.runs.length, 3)
  const report = { observed_failed_calls: 2, exact_payload_replays_succeeded_after_fix: 2,
    existing_run_ids_preserved: true, total_runs_after_replays: 3, new_model_calls: 0 }
  await writeFile('docs/pilot-regression-results.json', JSON.stringify(report, null, 2) + '\n')
  console.log(JSON.stringify(report))
} finally { await ctx.fiber.dispose() }
