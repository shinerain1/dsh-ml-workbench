/** Bounded JSON transport to the locally installed Python MLflow backend. */
import { spawn } from 'node:child_process'

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json }
export type JsonObject = { [key: string]: Json }

/** Host-supplied DSH identity; never taken from model arguments. */
export interface DshInvocation { session_id: string; call_id: string }

export interface ClientConfig {
  pythonExecutable: string
  root: string
  allowCustomCode: boolean
  requestTimeoutMs: number
  maxResponseBytes: number
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isJson(value: unknown): value is Json {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true
  if (typeof value === 'number') return Number.isFinite(value)
  if (Array.isArray(value)) return value.every(isJson)
  return isObject(value) && Object.values(value).every(isJson)
}

/** Owns only short control processes; persisted training workers outlive it. */
export class BackendClient {
  private readonly lifetime = new AbortController()
  private readonly active = new Set<Promise<JsonObject>>()

  constructor(private readonly config: ClientConfig) {}

  /** Send one versioned request, preserving caller cancellation and byte limits. */
  call(operation: string, args: unknown, signal: AbortSignal, dshContext?: DshInvocation): Promise<JsonObject> {
    const fused = AbortSignal.any([signal, this.lifetime.signal, AbortSignal.timeout(this.config.requestTimeoutMs)])
    const tracked = this.request(operation, args, fused, dshContext).finally(() => this.active.delete(tracked))
    this.active.add(tracked)
    return tracked
  }

  /** Abort control processes and wait until their handles have closed. */
  async dispose(): Promise<void> {
    this.lifetime.abort(new Error('ML workbench plugin unloaded'))
    await Promise.allSettled([...this.active])
  }

  private async request(operation: string, args: unknown, signal: AbortSignal, dshContext?: DshInvocation): Promise<JsonObject> {
    signal.throwIfAborted()
    const request = JSON.stringify({ protocol_version: 1, operation, args, dsh_context: dshContext })
    if (Buffer.byteLength(request) > 1024 * 1024) throw new Error('REQUEST_TOO_LARGE: RPC input exceeds 1 MiB')
    const argv = ['-m', 'dsh_ml_workbench', 'rpc', '--root', this.config.root]
    if (this.config.allowCustomCode) argv.push('--allow-custom-code')
    const raw = await new Promise<string>((resolve, reject) => {
      const child = spawn(this.config.pythonExecutable, argv, {
        shell: false, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
        env: { ...process.env, PYTHONUTF8: '1', MLFLOW_ENABLE_TELEMETRY: 'false' },
      })
      const chunks: Buffer[] = []
      let bytes = 0
      let stderr = Buffer.alloc(0)
      let failure: Error | undefined
      const abort = () => {
        const timeout = signal.reason instanceof Error && signal.reason.name === 'TimeoutError'
        failure = new Error(`${timeout ? 'REQUEST_TIMEOUT' : 'REQUEST_ABORTED'}: control request stopped; its outcome may already be committed. Use ml_workbench_resume or ml_experiment_list to recover IDs, then ml_experiment_get. Retry creation only with the same request_key or exact DSH call identity.`)
        child.kill()
      }
      signal.addEventListener('abort', abort, { once: true })
      if (signal.aborted) abort()
      child.stdout.on('data', (chunk: Buffer) => {
        bytes += chunk.length
        if (bytes > this.config.maxResponseBytes) {
          failure = new Error('RESPONSE_TOO_LARGE: backend output exceeded its configured limit')
          child.kill()
        } else chunks.push(chunk)
      })
      child.stderr.on('data', (chunk: Buffer) => { stderr = Buffer.concat([stderr, chunk]).subarray(-8192) })
      child.on('error', error => { failure = error })
      child.stdin.on('error', error => { failure ??= error })
      child.on('close', code => {
        signal.removeEventListener('abort', abort)
        if (failure !== undefined) { reject(failure); return }
        if (code !== 0) {
          reject(new Error(`BACKEND_UNAVAILABLE: Python exited with ${code}. Install the backend in pythonExecutable's environment. ${stderr.toString('utf8').slice(-2000)}`))
          return
        }
        resolve(Buffer.concat(chunks).toString('utf8'))
      })
      child.stdin.end(request)
    })
    const envelope: unknown = JSON.parse(raw)
    if (!isObject(envelope) || envelope.protocol_version !== 1 || typeof envelope.ok !== 'boolean') {
      throw new Error('INVALID_BACKEND_RESPONSE: expected protocol version 1')
    }
    if (!envelope.ok) {
      if (!isObject(envelope.error) || typeof envelope.error.code !== 'string' || typeof envelope.error.message !== 'string') {
        throw new Error('INVALID_BACKEND_RESPONSE: malformed failure')
      }
      throw new Error(`${envelope.error.code}: ${envelope.error.message}`)
    }
    if (!isObject(envelope.result) || !isJson(envelope.result)) throw new Error('INVALID_BACKEND_RESPONSE: result must be a JSON object')
    return envelope.result
  }
}
