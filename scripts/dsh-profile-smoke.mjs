/** Load a public installation with DSH's real profile resolver and host-provided SDK peers. */
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const checkout = process.env.DSH_CHECKOUT
const profileDir = process.env.MLWB_PROFILE
if (!checkout || !profileDir || !process.env.MLWB_PYTHON) throw new Error('Set DSH_CHECKOUT, MLWB_PROFILE and MLWB_PYTHON')
const load = path => import(pathToFileURL(join(checkout, path)).href)
const { Context } = await load('vendor/cordis/lib/index.js')
const { createRuntimeResolution, loadProfileDirectory, PluginPackages } = await load('packages/boot/app-boot/lib/index.js')
const anchor = join(checkout, 'apps/cli/package.json')
const profile = loadProfileDirectory('mlwb-profile-check', profileDir, anchor)
assert.deepEqual(profile.skippedBundles, [])
const resolution = await createRuntimeResolution({ installAnchor: anchor, profile,
  home: resolve(profileDir, '../..') })
const ctx = new Context()
try {
  await ctx.plugin(PluginPackages, { resolution })
  const require = createRequire(pathToFileURL(join(profileDir, 'package.json')))
  const pluginPath = require.resolve('dsh-ml-workbench')
  assert.ok(pluginPath.toLowerCase().includes('dsh-public-v4'))
  const pluginRequire = createRequire(pluginPath)
  for (const peer of ['@deepseek-ai/cordis', '@deepseek-ai/dsh-tools']) {
    assert.ok(pluginRequire.resolve(peer).toLowerCase().startsWith(resolve(checkout).toLowerCase()), peer)
  }
  const Plugin = await import(pathToFileURL(pluginPath).href)
  await ctx.plugin((await load('packages/core/system-prompt/lib/index.js')).default)
  await ctx.plugin((await load('packages/core/tools/lib/index.js')).default)
  await ctx.plugin(await load('packages/core/tool-executor-native/lib/index.js'))
  const plugin = ctx.plugin(Plugin, Plugin.Config({ root: resolve('.verification/public-profile-records-v4'),
    pythonExecutable: process.env.MLWB_PYTHON }))
  await plugin
  assert.equal(ctx.tools.schemas().filter(tool => tool.name.startsWith('ml_')).length, 20)
  const result = await ctx.tools.execute({ calls: [{ type: 'tool-call', id: 'profile-health', name: 'ml_workbench_health', arguments: '{}' }], signal: new AbortController().signal })
  assert.equal(result.results[0].result.isError, false, JSON.stringify(result))
  assert.equal(result.results[0].result.value.version, '0.4.0')
  await plugin.dispose()
  assert.equal(ctx.tools.schemas().filter(tool => tool.name.startsWith('ml_')).length, 0)
  console.log(JSON.stringify({ public_profile_resolution: 'passed', host_sdk_peers: 'passed', native_tools: 20,
    installed_backend: '0.4.0', health: 'passed', unload: 'passed' }))
} finally { await ctx.fiber.dispose() }
