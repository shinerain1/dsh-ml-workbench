import assert from 'node:assert/strict'
import { randomUUID } from 'node:crypto'
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import test from 'node:test'
import { Context } from '@deepseek-ai/cordis'
import Tools from '@deepseek-ai/dsh-tools'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import * as Plugin from '../lib/index.js'
import { BackendClient } from '../lib/client.js'

const python = process.env.MLWB_PYTHON ?? resolve('.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python')

async function fixture(t) {
  await mkdir('.verification', { recursive: true })
  const directory = await mkdtemp(resolve('.verification', 'node-'))
  const ctx = new Context()
  t.after(() => ctx.fiber.dispose())
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(Tools)
  const plugin = ctx.plugin(Plugin, Plugin.Config({ pythonExecutable: python, root: join(directory, 'records') }))
  await plugin
  async function call(name, args = {}) {
    const result = await ctx.tools.execute({ callId: randomUUID(), name, arguments: args, signal: new AbortController().signal })
    assert.equal(result.isError, false, JSON.stringify(result))
    return result.value
  }
  return { ctx, plugin, directory, call }
}

test('real DSH tool registry drives the Python MLflow lifecycle and unload removes tools', { timeout: 300000 }, async t => {
  const { ctx, plugin, directory, call } = await fixture(t)
  assert.equal(ctx.tools.schemas().filter(tool => tool.name.startsWith('ml_')).length, 9)
  const path = join(directory, 'customers.csv')
  await writeFile(path, 'spend,visits,channel,target\n' + Array.from({ length: 80 }, (_, i) => `${i % 20},${i % 8},${i % 2 ? 'web' : 'shop'},${i % 4 > 1 ? 1 : 0}`).join('\n'))
  const created = await call('ml_experiment_create', { dataset_path: path, target: 'target', task: 'classification', model: 'logistic_regression', request_key: 'node-demo' })
  const started = await call('ml_experiment_start', { run_id: created.run_id })
  assert.ok(['starting', 'running', 'succeeded'].includes(started.state))
  let finished
  const deadline = Date.now() + 120000
  while (Date.now() < deadline) {
    finished = await call('ml_experiment_get', { run_id: created.run_id })
    if (['succeeded', 'failed', 'interrupted'].includes(finished.state)) break
    await delay(100)
  }
  assert.equal(finished.state, 'succeeded', JSON.stringify(finished))
  assert.equal(typeof finished.metrics.accuracy, 'number')
  const exported = await call('ml_experiment_export', { run_id: created.run_id })
  assert.equal(exported.data_included, false)
  assert.ok(exported.path.endsWith('.zip'))
  const invalid = await ctx.tools.execute({ callId: randomUUID(), name: 'ml_experiment_start', arguments: { run_id: 123 }, signal: new AbortController().signal })
  assert.equal(invalid.isError, true)
  await plugin.dispose()
  assert.equal(ctx.tools.schemas().filter(tool => tool.name.startsWith('ml_')).length, 0)
})

test('a pre-aborted control request never starts a Python process', async () => {
  const client = new BackendClient({ pythonExecutable: 'this-executable-does-not-exist', root: resolve('.verification'), allowCustomCode: false, requestTimeoutMs: 1000, maxResponseBytes: 4096 })
  await assert.rejects(client.call('health', {}, AbortSignal.abort(new Error('user cancelled'))), /user cancelled/)
  await client.dispose()
})

test('a missing Python executable is surfaced as a failure', async () => {
  const client = new BackendClient({ pythonExecutable: 'this-executable-does-not-exist', root: resolve('.verification'), allowCustomCode: false, requestTimeoutMs: 1000, maxResponseBytes: 4096 })
  await assert.rejects(client.call('health', {}, new AbortController().signal), /ENOENT/)
  await client.dispose()
})
