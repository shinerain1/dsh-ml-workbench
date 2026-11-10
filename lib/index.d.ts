import type { Context } from '@deepseek-ai/cordis';
import z from '@deepseek-ai/schemastery';
export declare const name = "ml-workbench";
export declare const inject: string[];
/** Host-local backend settings. Use one root per independent project or team. */
export interface Config {
    pythonExecutable: string;
    root: string;
    allowCustomCode: boolean;
    requestTimeoutMs: number;
    maxResponseBytes: number;
    executionMode: 'local' | 'dsh';
}
export declare const Config: z<Config>;
/** Register reversible tools on the DSH tool service and own control-process cleanup. */
export declare function apply(ctx: Context, config: Config): void;
