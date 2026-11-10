import assert from 'node:assert/strict'
import test from 'node:test'
import { ConfinedBackend } from '../lib/confined.js'

test('DSH execution fails closed when its confinement services are absent', async () => {
  const backend = new ConfinedBackend({ get: () => undefined }, { pythonExecutable: 'must-not-spawn' })
  assert.throws(() => backend.transport(undefined), /DSH_SANDBOX_UNAVAILABLE.*no local fallback/)
  await backend.dispose()
})

test('confinement failure cannot reach subprocess spawn or widen writable roots', async () => {
  let spawned = 0
  const policy = { mode: 'workspace-write', workspaceRoot: 'C:/workspace', additionalWritableRoots: [] }
  const backend = new ConfinedBackend({ get: name => ({
    sandboxPolicy: { resolve: () => policy },
    sandbox: { confine: async (_argv, actual) => { assert.equal(actual, policy); throw new Error('runner unavailable') } },
    subprocess: { resolveExecutable: async () => 'C:/python.exe', spawn: () => spawned++ },
  })[name] }, { pythonExecutable: 'python', root: 'C:/outside' })
  await assert.rejects(backend.transport(undefined).request([], '{}', new AbortController().signal), /runner unavailable/)
  assert.equal(spawned, 0)
  assert.deepEqual(policy.additionalWritableRoots, [])
  await backend.dispose()
})
