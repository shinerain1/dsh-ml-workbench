/** DSH tools for MLflow experiments; model execution is owned by the Python backend. */
import { homedir } from 'node:os';
import { isAbsolute, join } from 'node:path';
import z from '@deepseek-ai/schemastery';
import { defineTool } from '@deepseek-ai/dsh-tools';
import { BackendClient } from './client.js';
export const name = 'ml-workbench';
export const inject = ['tools'];
export const Config = z.object({
    pythonExecutable: z.string().default('python').description('Python interpreter with dsh-ml-workbench installed'),
    root: z.string().default(join(homedir(), '.dsh-ml-workbench')).description('Absolute path for local MLflow records and immutable snapshots'),
    allowCustomCode: z.boolean().default(false).description('Allow reviewed local Python preprocessing and estimator factories'),
    requestTimeoutMs: z.number().min(1000).max(2147483647).default(120000),
    maxResponseBytes: z.number().min(1024).max(16777216).default(4194304),
});
const runId = { type: 'string', required: true, description: 'MLflow run ID returned by create or branch' };
const requestKey = { type: 'string', description: 'Stable unique key for this intended creation; reuse only when retrying the same request' };
const models = ['logistic_regression', 'ridge', 'random_forest', 'extra_trees', 'hist_gradient_boosting', 'custom'];
const preprocessing = {
    type: 'object', additionalProperties: false, properties: {
        numeric_imputer: { type: 'string', enum: ['median', 'mean', 'most_frequent'] },
        scale_numeric: { type: 'boolean' },
        max_categories: { type: 'integer' },
        drop_columns: { type: 'array', items: { type: 'string' } },
    },
};
const tools = [
    { operation: 'health', name: 'ml_workbench_health', description: 'Check Python, MLflow, dependency versions and the local experiment root before training.', parameters: {} },
    { operation: 'create', name: 'ml_experiment_create',
        description: 'Snapshot a local UTF-8 CSV, code, parameters, environment and a fixed validation split into an MLflow experiment. Does not train. Exclude IDs and leakage columns explicitly. Only select classification or regression when labels are known. Custom factories require allowCustomCode.',
        parameters: {
            dataset_path: { type: 'string', required: true, description: 'Absolute path to a local CSV (maximum 128 MiB)' },
            target: { type: 'string', required: true }, task: { type: 'string', required: true, enum: ['classification', 'regression'] },
            name: { type: 'string' }, experiment_name: { type: 'string' }, model: { type: 'string', enum: models },
            params: { type: 'object', additionalProperties: true, description: 'scikit-learn estimator arguments; seed is separate and n_jobs is fixed at 1' },
            preprocessing, seed: { type: 'integer' }, test_fraction: { type: 'number' },
            preprocessor_path: { type: 'string', description: 'Reviewed .py defining make_preprocessor(X_train, config)' },
            estimator_path: { type: 'string', description: 'Reviewed .py defining make_estimator(task, params, seed); requires model=custom' },
            request_key: requestKey,
        } },
    { operation: 'start', name: 'ml_experiment_start',
        description: 'Start one detached training worker. Repeated calls return the same run and never launch it twice. Query get until terminal. Closing the DSH session leaves training running.', parameters: { run_id: runId } },
    { operation: 'get', name: 'ml_experiment_get',
        description: 'Read durable state, lifecycle events, bounded log tail and verified metrics. Missing workers become interrupted. A failed/interrupted fit must be restarted as a new branch; weights are not resumed.', parameters: { run_id: runId } },
    { operation: 'list', name: 'ml_experiment_list',
        description: 'Recover recent run IDs after a session restart. Returns newest runs first.', parameters: { limit: { type: 'integer', description: '1 to 100; default 20' } } },
    { operation: 'cancel', name: 'ml_experiment_cancel',
        description: 'Cancel a created or running experiment, checking process identity and waiting for termination. Completed runs are unchanged.', parameters: { run_id: runId } },
    { operation: 'branch', name: 'ml_experiment_branch',
        description: 'Create a new immutable experiment from a previous run. Default reuse_preprocessing=true copies its fitted preprocessor; parent must have succeeded and data, split and preprocessing must stay identical. Set false to change preprocessing or restart a failed/interrupted run. changes replaces specified config fields; dataset/target/task cannot change.',
        parameters: { run_id: runId, changes: { type: 'object', additionalProperties: true, description: 'Experiment field overrides, e.g. {"name":"faster","model":"logistic_regression","params":{}}' },
            reuse_preprocessing: { type: 'boolean' }, request_key: requestKey } },
    { operation: 'compare', name: 'ml_experiment_compare',
        description: 'Rank 2–30 successful runs only when data, target, validation rows and evaluator match. Refuses incompatible conditions. Timing is one local measurement, not a statistical speed claim.',
        parameters: { run_ids: { type: 'array', required: true, items: { type: 'string' } }, metric: { type: 'string', enum: ['accuracy', 'balanced_accuracy', 'f1_macro', 'rmse', 'mae', 'r2'] } } },
    { operation: 'export', name: 'ml_experiment_export',
        description: 'Verify code/model/MLflow consistency and create a local ZIP with model, exact training recipe, report, pinned environment and standalone reproduce.py. CSV data is excluded unless include_data=true. Does not upload anything.',
        parameters: { run_id: runId, include_data: { type: 'boolean', description: 'Include the snapshotted CSV in this local delivery archive; default false' } } },
];
/** Register reversible tools on the DSH tool service and own control-process cleanup. */
export function apply(ctx, config) {
    if (!isAbsolute(config.root))
        throw new Error('ml-workbench: root must be an absolute local path');
    if (!config.pythonExecutable.trim())
        throw new Error('ml-workbench: pythonExecutable is required');
    if (!Number.isInteger(config.requestTimeoutMs) || config.requestTimeoutMs < 1000 || config.requestTimeoutMs > 2147483647)
        throw new Error('ml-workbench: invalid requestTimeoutMs');
    if (!Number.isInteger(config.maxResponseBytes) || config.maxResponseBytes < 1024 || config.maxResponseBytes > 16777216)
        throw new Error('ml-workbench: invalid maxResponseBytes');
    const backend = new BackendClient(config);
    ctx.effect(() => () => backend.dispose());
    for (const tool of tools) {
        ctx.tools.register(defineTool({
            name: tool.name, description: tool.description, parameters: tool.parameters,
            output: {
                schema: { type: 'object', additionalProperties: true },
                render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
            },
            execute: (args, exec) => backend.call(tool.operation, args, exec.signal),
            presentCall: args => ({ card: 'generic', title: tool.name, rawInput: args }),
        }));
    }
}
