export type Json = null | boolean | number | string | Json[] | {
    [key: string]: Json;
};
export type JsonObject = {
    [key: string]: Json;
};
/** Host-supplied DSH identity; never taken from model arguments. */
export interface DshInvocation {
    session_id: string;
    call_id: string;
}
export interface ClientConfig {
    pythonExecutable: string;
    root: string;
    allowCustomCode: boolean;
    requestTimeoutMs: number;
    maxResponseBytes: number;
}
export interface RpcTransport {
    request(argv: string[], input: string, signal: AbortSignal): Promise<string>;
    launch(runId: string, token: string, signal: AbortSignal): Promise<void>;
}
/** Owns only short control processes; persisted training workers outlive it. */
export declare class BackendClient {
    private readonly config;
    private readonly lifetime;
    private readonly active;
    constructor(config: ClientConfig);
    /** Send one versioned request, preserving caller cancellation and byte limits. */
    call(operation: string, args: unknown, signal: AbortSignal, dshContext?: DshInvocation, transport?: RpcTransport): Promise<JsonObject>;
    /** Abort control processes and wait until their handles have closed. */
    dispose(): Promise<void>;
    private request;
}
