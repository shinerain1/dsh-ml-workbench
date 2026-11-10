/** DSH-owned file confinement and process ranges. No fallback to host spawning. */
import type { Context } from '@deepseek-ai/cordis';
import type { ClientConfig, RpcTransport } from './client.js';
export declare class ConfinedBackend {
    private readonly ctx;
    private readonly config;
    private readonly workers;
    private readonly lifetime;
    constructor(ctx: Context, config: ClientConfig);
    transport(session: unknown): RpcTransport;
    dispose(): Promise<void>;
}
