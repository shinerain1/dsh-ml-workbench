import assert from 'node:assert/strict'
import { mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import test from 'node:test'
import { BackendClient } from '../lib/client.js'

test('a timed-out control call reports an unknown committed outcome and releases its process', { timeout: 20000 }, async () => {
  await mkdir('.verification', { recursive: true })
  const folder = await mkdtemp(resolve('.verification/transport-'))
  const module = join(folder, 'dsh_ml_workbench')
  await mkdir(module)
  await writeFile(join(module, '__init__.py'), '')
  await writeFile(join(module, '__main__.py'), `import json, os, sys, time
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--root') + 1])
json.loads(sys.stdin.read())
(root / 'committed.json').write_text(json.dumps({'pid': os.getpid()}))
time.sleep(12)
print(json.dumps({'protocol_version': 1, 'ok': True, 'result': {}}))
`)
  const old = process.env.PYTHONPATH
  process.env.PYTHONPATH = folder
  const client = new BackendClient({ pythonExecutable: process.env.MLWB_PYTHON ?? resolve('.venv/Scripts/python.exe'),
    root: folder, allowCustomCode: false, requestTimeoutMs: 2000, maxResponseBytes: 4096 })
  try {
    const start = Date.now()
    await assert.rejects(client.call('create', {}, new AbortController().signal), /REQUEST_TIMEOUT.*already be committed.*ml_workbench_resume/)
    assert.ok(Date.now() - start < 8000, 'The timeout must not wait for the original 12-second operation')
    const { pid } = JSON.parse(await readFile(join(folder, 'committed.json'), 'utf8'))
    assert.throws(() => process.kill(pid, 0), { code: 'ESRCH' })
  } finally {
    await client.dispose()
    if (old === undefined) delete process.env.PYTHONPATH
    else process.env.PYTHONPATH = old
  }
})
