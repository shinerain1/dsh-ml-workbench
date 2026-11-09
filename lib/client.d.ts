export type Json = null | boolean | number | string | Json[] | {
    [key: string]: Json;
};
export type JsonObject = {
    [key: string]: Json;
};
export interface ClientConfig {
    pythonExecutable: string;
    root: string;
    allowCustomCode: boolean;
    requestTimeoutMs: number;
    maxResponseBytes: number;
}
/** Owns only short control processes; persisted training workers outlive it. */
export declare class BackendClient {
    private readonly config;
    private readonly lifetime;
    private readonly active;
    constructor(config: ClientConfig);
    /** Send one versioned request, preserving caller cancellation and byte limits. */
    call(operation: string, args: unknown, signal: AbortSignal): Promise<JsonObject>;
    /** Abort control processes and wait until their handles have closed. */
    dispose(): Promise<void>;
    private request;
}
