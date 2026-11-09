/** Exercise the built plugin against the tool executor in an actual DSH checkout. */
import assert from 'node:assert/strict'
import { mkdir } from 'node:fs/promises'
import { resolve, join } from 'node:path'
import { pathToFileURL } from 'node:url'

const checkout = process.env.DSH_CHECKOUT
if (!checkout) throw new Error('Set DSH_CHECKOUT to a built DeepSeek Harness checkout')
const load = path => import(pathToFileURL(join(checkout, path)).href)
const Plugin = await import(process.env.MLWB_PLUGIN
  ? pathToFileURL(resolve(process.env.MLWB_PLUGIN)).href : new URL('../lib/index.js', import.meta.url).href)
const { Context } = await load('vendor/cordis/lib/index.js')
const { default: Tools } = await load('packages/core/tools/lib/index.js')
const { default: SystemPrompt } = await load('packages/core/system-prompt/lib/index.js')
const Native = await load('packages/core/tool-executor-native/lib/index.js')
const root = resolve('.verification/dsh-source-smoke')
await mkdir(root, { recursive: true })
const ctx = new Context()
try {
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(Tools)
  await ctx.plugin(Native)
  const plugin = ctx.plugin(Plugin, Plugin.Config({ root, pythonExecutable: process.env.MLWB_PYTHON ?? resolve('.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python') }))
  await plugin
  const names = ctx.tools.schemas().map(tool => tool.name)
  assert.equal(names.filter(name => name.startsWith('ml_')).length, 17)
  const result = await ctx.tools.execute({ calls: [{ type: 'tool-call', id: 'mlwb-smoke', name: 'ml_workbench_health', arguments: '{}' }], signal: new AbortController().signal })
  assert.equal(result.results[0].result.isError, false, JSON.stringify(result))
  assert.equal(result.results[0].result.value.protocol_version, 1)
  await plugin.dispose()
  assert.equal(ctx.tools.schemas().filter(tool => tool.name.startsWith('ml_')).length, 0)
  console.log(JSON.stringify({ dsh_tools: 'passed', tool_count: 17, protocol_version: 1, unload: 'passed' }))
} finally { await ctx.fiber.dispose() }
