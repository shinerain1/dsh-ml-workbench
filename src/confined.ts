/** DSH-owned file confinement and process ranges. No fallback to host spawning. */
import type { Context } from '@deepseek-ai/cordis'
import type { ClientConfig, RpcTransport } from './client.js'

interface Policy { mode: 'read-only' | 'workspace-write' | 'danger-full-access'; workspaceRoot: string; additionalWritableRoots?: readonly string[]; sessionId?: string }
interface Handle {
  done: Promise<{ exitCode: number | null }>
  collected: { stdout?: { readFrom(offset: number): { text: string; lossy: boolean } }; stderr?: { readFrom(offset: number): { text: string; lossy: boolean } } }
  terminate(): void
  waitForExit(): Promise<void>
}
interface Subprocess {
  resolveExecutable(command: string, env?: Record<string, string>, signal?: AbortSignal): Promise<string>
  spawn(spec: { argv: readonly string[]; cwd: string; stdio: { stdin: { data: string } | 'ignore'; stdout: { maxBytes: number }; stderr: { maxBytes: number } }; graceMs: number; signal?: AbortSignal; env: Record<string, string> }): Handle
}

export class ConfinedBackend {
  private readonly workers = new Set<Handle>()
  private readonly lifetime = new AbortController()
  constructor(private readonly ctx: Context, private readonly config: ClientConfig) {}

  transport(session: unknown): RpcTransport {
    const sandbox = this.ctx.get('sandbox') as { confine(argv: readonly string[], policy: Policy, signal?: AbortSignal): Promise<{ argv: string[]; enforcement: string }> } | undefined
    const resolver = this.ctx.get('sandboxPolicy') as { resolve(request: { session: unknown }): Policy } | undefined
    const subprocess = this.ctx.get('subprocess') as Subprocess | undefined
    if (!sandbox || !resolver || !subprocess) throw new Error('DSH_SANDBOX_UNAVAILABLE: executionMode=dsh requires sandbox, sandboxPolicy and subprocess services; no local fallback')
    const policy = resolver.resolve({ session })
    const env = { PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1', MLFLOW_ENABLE_TELEMETRY: 'false' }
    const wrap = async (args: string[], signal: AbortSignal) => {
      signal.throwIfAborted()
      const python = await subprocess.resolveExecutable(this.config.pythonExecutable, env, signal)
      const argv = [python, ...args]
      return policy.mode === 'danger-full-access' ? argv : (await sandbox.confine(argv, policy, signal)).argv
    }
    return {
      request: async (args, input, signal) => {
        const argv = await wrap(args, signal)
        const handle = subprocess.spawn({ argv, cwd: policy.workspaceRoot,
          stdio: { stdin: { data: input }, stdout: { maxBytes: this.config.maxResponseBytes }, stderr: { maxBytes: 8192 } },
          graceMs: 3000, signal, env })
        try {
          const outcome = await handle.done
          if (signal.aborted) throw new Error('REQUEST_ABORTED: confined control request stopped; use resume/get before retrying')
          const stdout = handle.collected.stdout!.readFrom(0)
          if (stdout.lossy) throw new Error('RESPONSE_TOO_LARGE: confined backend output exceeded limit')
          if (outcome.exitCode !== 0) throw new Error(`DSH_BACKEND_FAILED: exit ${outcome.exitCode}; ${handle.collected.stderr?.readFrom(0).text.slice(-2000)}`)
          return stdout.text
        } finally {
          handle.terminate()
          await handle.waitForExit()
        }
      },
      launch: async (runId, token, signal) => {
        const argv = await wrap(['-m', 'dsh_ml_workbench', 'supervise', '--root', this.config.root, '--run-id', runId, '--token', token], signal)
        this.lifetime.signal.throwIfAborted()
        const handle = subprocess.spawn({ argv, cwd: policy.workspaceRoot,
          stdio: { stdin: 'ignore', stdout: { maxBytes: 4096 }, stderr: { maxBytes: 8192 } },
          graceMs: 3000, signal: this.lifetime.signal, env })
        this.workers.add(handle)
        // A committed launch outlives the initiating tool call, but never the plugin owner.
        void handle.done.catch(() => undefined).finally(async () => {
          handle.terminate()
          try { await handle.waitForExit() } finally { this.workers.delete(handle) }
        }).catch(() => undefined)
      },
    }
  }

  async dispose(): Promise<void> {
    this.lifetime.abort()
    const workers = [...this.workers]
    workers.forEach(worker => worker.terminate())
    const results = await Promise.allSettled(workers.map(worker => worker.waitForExit()))
    const failure = results.find(result => result.status === 'rejected')
    if (failure?.status === 'rejected') throw failure.reason
  }
}
