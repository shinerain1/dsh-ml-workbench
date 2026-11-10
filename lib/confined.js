export class ConfinedBackend {
    ctx;
    config;
    workers = new Set();
    lifetime = new AbortController();
    constructor(ctx, config) {
        this.ctx = ctx;
        this.config = config;
    }
    transport(session) {
        const sandbox = this.ctx.get('sandbox');
        const resolver = this.ctx.get('sandboxPolicy');
        const subprocess = this.ctx.get('subprocess');
        if (!sandbox || !resolver || !subprocess)
            throw new Error('DSH_SANDBOX_UNAVAILABLE: executionMode=dsh requires sandbox, sandboxPolicy and subprocess services; no local fallback');
        const policy = resolver.resolve({ session });
        const env = { PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1', MLFLOW_ENABLE_TELEMETRY: 'false' };
        const wrap = async (args, signal) => {
            signal.throwIfAborted();
            const python = await subprocess.resolveExecutable(this.config.pythonExecutable, env, signal);
            const argv = [python, ...args];
            return policy.mode === 'danger-full-access' ? argv : (await sandbox.confine(argv, policy, signal)).argv;
        };
        return {
            request: async (args, input, signal) => {
                const argv = await wrap(args, signal);
                const handle = subprocess.spawn({ argv, cwd: policy.workspaceRoot,
                    stdio: { stdin: { data: input }, stdout: { maxBytes: this.config.maxResponseBytes }, stderr: { maxBytes: 8192 } },
                    graceMs: 3000, signal, env });
                try {
                    const outcome = await handle.done;
                    if (signal.aborted)
                        throw new Error('REQUEST_ABORTED: confined control request stopped; use resume/get before retrying');
                    const stdout = handle.collected.stdout.readFrom(0);
                    if (stdout.lossy)
                        throw new Error('RESPONSE_TOO_LARGE: confined backend output exceeded limit');
                    if (outcome.exitCode !== 0)
                        throw new Error(`DSH_BACKEND_FAILED: exit ${outcome.exitCode}; ${handle.collected.stderr?.readFrom(0).text.slice(-2000)}`);
                    return stdout.text;
                }
                finally {
                    handle.terminate();
                    await handle.waitForExit();
                }
            },
            launch: async (runId, token, signal) => {
                const argv = await wrap(['-m', 'dsh_ml_workbench', 'supervise', '--root', this.config.root, '--run-id', runId, '--token', token], signal);
                this.lifetime.signal.throwIfAborted();
                const handle = subprocess.spawn({ argv, cwd: policy.workspaceRoot,
                    stdio: { stdin: 'ignore', stdout: { maxBytes: 4096 }, stderr: { maxBytes: 8192 } },
                    graceMs: 3000, signal: this.lifetime.signal, env });
                this.workers.add(handle);
                // A committed launch outlives the initiating tool call, but never the plugin owner.
                void handle.done.catch(() => undefined).finally(async () => {
                    handle.terminate();
                    try {
                        await handle.waitForExit();
                    }
                    finally {
                        this.workers.delete(handle);
                    }
                }).catch(() => undefined);
            },
        };
    }
    async dispose() {
        this.lifetime.abort();
        const workers = [...this.workers];
        workers.forEach(worker => worker.terminate());
        const results = await Promise.allSettled(workers.map(worker => worker.waitForExit()));
        const failure = results.find(result => result.status === 'rejected');
        if (failure?.status === 'rejected')
            throw failure.reason;
    }
}
