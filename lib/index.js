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
const requestKey = { type: 'string', description: 'Stable unique key for this intended creation across new calls or PTC reruns. If omitted in a DSH Session, exact replay of the same call ID is deduplicated automatically.' };
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
    { operation: 'resume', name: 'ml_workbench_resume',
        description: 'Recover experiment and Optuna study references linked to this DSH Session after restart or compaction. No training is started. Follow each returned next_tool to verify current results. Use previous_session_id to continue another known local Session; this is navigation, not access control. Legacy runs without Session links remain in ml_experiment_list.',
        parameters: { previous_session_id: { type: 'string' }, limit: { type: 'integer', description: '1-100; default 20' }, offset: { type: 'integer' } } },
    { operation: 'create', name: 'ml_experiment_create',
        description: 'Snapshot a local UTF-8 CSV, code, parameters, environment and a fixed validation split into an MLflow experiment. Does not train. Exclude IDs and leakage columns explicitly. Only select classification or regression when labels are known. Custom factories require allowCustomCode.',
        parameters: {
            dataset_path: { type: 'string', required: true, description: 'Absolute path to a local CSV (maximum 128 MiB)' },
            target: { type: 'string', required: true }, task: { type: 'string', required: true, enum: ['classification', 'regression'] },
            name: { type: 'string' }, experiment_name: { type: 'string' }, model: { type: 'string', enum: models },
            params: { type: 'object', additionalProperties: true, description: 'scikit-learn estimator arguments. seed controls randomness and n_jobs is fixed at 1; redundant random_state matching seed and n_jobs=1 are accepted and normalized. Conflicts are rejected.' },
            preprocessing, seed: { type: 'integer' }, test_fraction: { type: 'number' },
            split: { type: 'object', additionalProperties: false, properties: {
                    strategy: { type: 'string', enum: ['random', 'group', 'time'] },
                    column: { type: 'string', description: 'Required for group/time. Excluded from model features. Time requires ISO date/time strings.' },
                } },
            preprocessor_path: { type: 'string', description: 'Reviewed .py defining make_preprocessor(X_train, config)' },
            estimator_path: { type: 'string', description: 'Reviewed .py defining make_estimator(task, params, seed); requires model=custom' },
            request_key: requestKey,
        } },
    { operation: 'start', name: 'ml_experiment_start',
        description: 'Start one detached training worker. Repeated calls return the same run and never launch it twice. Query get until terminal. Closing the DSH session leaves training running.', parameters: { run_id: runId } },
    { operation: 'get', name: 'ml_experiment_get',
        description: 'Read durable state, lifecycle events, bounded log tail and verified metrics. Missing workers become interrupted. A failed/interrupted fit must be restarted as a new branch; weights are not resumed.', parameters: { run_id: runId } },
    { operation: 'list', name: 'ml_experiment_list',
        description: 'Recover run IDs after a session restart. Newest first, optional filters and offset pagination. next_offset=null means no more results.', parameters: {
            limit: { type: 'integer', description: '1 to 100; default 20' }, offset: { type: 'integer' },
            state: { type: 'string', enum: ['created', 'starting', 'running', 'cancelling', 'succeeded', 'failed', 'interrupted', 'cancelled'] },
            model: { type: 'string' }, parent_run_id: { type: 'string' },
        } },
    { operation: 'cancel', name: 'ml_experiment_cancel',
        description: 'Cancel a created or running experiment, checking process identity and waiting for termination. Completed runs are unchanged.', parameters: { run_id: runId } },
    { operation: 'branch', name: 'ml_experiment_branch',
        description: 'Create a new immutable experiment from a previous run. Default reuse_preprocessing=true copies its fitted preprocessor; parent must have succeeded and data, split and preprocessing must stay identical. Set false to change preprocessing or restart a failed/interrupted run. changes replaces specified config fields; dataset/target/task cannot change.',
        parameters: { run_id: runId, changes: { type: 'object', additionalProperties: true, description: 'Experiment field overrides, e.g. {"name":"faster","model":"logistic_regression","params":{}}' },
            reuse_preprocessing: { type: 'boolean' }, request_key: requestKey } },
    { operation: 'compare', name: 'ml_experiment_compare',
        description: 'Rank 2–30 successful runs only when data, target, validation rows and evaluator match. Refuses incompatible conditions. Timing is one local measurement, not a statistical speed claim.',
        parameters: { run_ids: { type: 'array', required: true, items: { type: 'string' } }, metric: { type: 'string', enum: ['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc', 'average_precision', 'log_loss', 'rmse', 'mae', 'r2'] } } },
    { operation: 'export', name: 'ml_experiment_export',
        description: 'Verify code/model/MLflow consistency and create a local ZIP with model, exact training recipe, report, pinned environment and standalone reproduce.py. CSV data is excluded unless include_data=true. Does not upload anything.',
        parameters: { run_id: runId, include_data: { type: 'boolean', description: 'Include the snapshotted CSV in this local delivery archive; default false' } } },
    { operation: 'inspect', name: 'ml_dataset_inspect',
        description: 'Inspect a local CSV before choosing features and evaluation split. Reports types, missing values, duplicates and possible identifiers without returning raw rows. Does not automatically detect leakage.',
        parameters: { dataset_path: { type: 'string', required: true }, target: { type: 'string' } } },
    { operation: 'diff', name: 'ml_experiment_diff',
        description: 'Explain config, code and evaluation differences between two immutable experiments, including whether their evaluation contracts match. Does not rank incompatible scores.',
        parameters: { left_run_id: { type: 'string', required: true }, right_run_id: { type: 'string', required: true } } },
    { operation: 'predict', name: 'ml_model_predict',
        description: 'Apply a verified successful MLflow pipeline to another local CSV. Validates required features and numeric types; writes predictions and class probabilities plus a manifest in a new local directory. Does not retrain. Raw input and predictions remain local.',
        parameters: { run_id: runId, dataset_path: { type: 'string', required: true } } },
    { operation: 'search_create', name: 'ml_search_create',
        description: 'Create a durable Optuna study from a built-in estimator run. study_name is an idempotent immutable identifier. search_space maps estimator parameter names to {type:int|float,low,high,log?,step?} or {type:categorical,choices:[...]}. Trial budget includes failures. Does not start training.',
        parameters: { study_name: { type: 'string', required: true }, run_id: runId,
            search_space: { type: 'object', required: true, additionalProperties: true },
            n_trials: { type: 'integer', description: 'Total budget 1-100; default 10' }, metric: { type: 'string' },
            seed: { type: 'integer' }, reuse_preprocessing: { type: 'boolean', description: 'Default false refits on the same training rows; true requires a successful parent' },
            trial_timeout_seconds: { type: 'integer', description: '5-86400, default 600. Enforced when step/get polls, not by a background timer.' },
        } },
    { operation: 'search_step', name: 'ml_search_step',
        description: 'Reconcile an Optuna study and start at most one trial; nonblocking and safe to retry. Poll step until complete/cancelled to advance the whole search. After DSH restart the same study_name resumes, with no duplicate active fit. Best validation score is not an independent final-test score.',
        parameters: { study_name: { type: 'string', required: true } } },
    { operation: 'search_get', name: 'ml_search_get',
        description: 'Read durable Optuna trials, linked MLflow runs and verified best run. Reconciles finished trials and expired fits but never starts a new fit. Use search_step to continue scheduling.',
        parameters: { study_name: { type: 'string', required: true } } },
    { operation: 'search_cancel', name: 'ml_search_cancel',
        description: 'Persistently stop scheduling a study and cancel its active fit with process-exit confirmation. Completed experiment results remain available. Use a new study_name to run another search.',
        parameters: { study_name: { type: 'string', required: true } } },
];
/** Typed handles are available to DSH PTC; extra domain fields remain canonical JSON. */
function outputSchema(operation) {
    const schema = { type: 'object', additionalProperties: true };
    if (['create', 'branch', 'start', 'get', 'cancel'].includes(operation)) {
        schema.properties = { run_id: runId, state: { type: 'string', required: true } };
    }
    else if (operation.startsWith('search_')) {
        schema.properties = { study_name: { type: 'string', required: true }, state: { type: 'string', required: true } };
    }
    else if (operation === 'resume') {
        schema.properties = {
            session_id: { type: 'string', required: true },
            next_offset: { required: true, oneOf: [{ type: 'integer' }, { type: 'null' }] },
            resources: { type: 'array', required: true, items: { type: 'object', additionalProperties: true, properties: {
                        kind: { type: 'string', required: true, enum: ['run', 'study'] },
                        ref: { type: 'string', required: true }, state: { type: 'string', required: true },
                        next_tool: { type: 'string', required: true }, arguments: { type: 'object', additionalProperties: true, required: true },
                    } } },
        };
    }
    return schema;
}
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
                schema: outputSchema(tool.operation),
                render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }],
            },
            execute: (args, exec) => backend.call(tool.operation, args, exec.signal, exec.agent === undefined ? undefined : { session_id: exec.agent.session.id, call_id: exec.callId }),
            presentCall: args => ({ card: 'generic', title: tool.name, rawInput: args }),
        }));
    }
}
