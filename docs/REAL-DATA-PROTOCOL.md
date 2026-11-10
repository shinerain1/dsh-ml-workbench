# Real-data workflow evaluation — protocol fixed before results

Date: 2026-11-10. Backend release: 0.4.0. This is a controlled workflow experiment,
not a randomized study of developer productivity or a new training algorithm.

## Tasks and source data

1. [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail),
   Chen (2015), DOI 10.24432/C5BW33, CC BY 4.0. Predict next-calendar-month
   repeat purchase among customers who purchased in the preceding 90 days.
   Monthly anchors: March through November 2011. Features use transactions
   strictly before the anchor; labels use [anchor, next month). Remove missing
   customer IDs, cancellations, nonpositive quantities/prices, and exact duplicate
   rows, recording sequential removal counts. Exclude customer ID and anchor
   from predictors. This estimates return of existing customers, not new-customer
   generalization. Repeated customers across time are intentional.
2. [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset),
   Fanaee-T (2013), DOI 10.24432/C5W894, CC BY 4.0. Estimate hourly rental count
   conditional on recorded calendar/weather features. Exclude casual and registered
   (their sum is the target), row ID and raw date; build a time-split timestamp.
   Realized weather is used, so this is not a deployable advance weather forecast.

Use the complete derived datasets, not a favorable sample. Store source/derived
SHA-256, row counts, feature definitions, and window audits. Raw data and customer
identifiers remain local in ignored directories; publish scripts and aggregates.

## Fixed modeling and selection

Time-ordered train/validation/final partitions, requested fractions 60/20/20.
Equal timestamps remain together, so actual sizes may differ. The training
period's label windows must end no later than the next partition's first anchor.
Preprocessing is median imputation and standard scaling fitted on training only.
Models for seeds 7, 19, 41: logistic regression (max_iter=1000) / ridge (alpha=1),
random forest (64 trees, min_samples_leaf=5), histogram gradient boosting
(80 iterations, max_leaf_nodes=15). Single-thread numerical libraries.

For each seed, branch the linear experiment to forest and boosting, preserving
the exact fitted preprocessing. Compare all nine candidates on the identical
time split. Select one candidate per dataset using validation average precision
(classification, maximize) or RMSE (regression, minimize). Finalize ONLY that
candidate. Report all validation results, dummy baselines, final scores, and
training versus orchestration times; do not choose models based on final scores.
Run one extra seed-7 forest with random partition as a *separate evaluation
question*. Report its validation score without ranking it against time-split runs
or examining its final holdout. Test that the plugin rejects mixed comparisons.

## Fair baseline and workflow checks

For every time-split candidate, directly run the same frozen sklearn recipe and
inputs, log code/data/split/environment/model/metrics using native MLflow, and
compare metrics. This intentionally strong, shared-recipe baseline isolates
workflow behavior. It is not independently authored Agent code. Neither arm is
expected to improve predictive accuracy over the other.

For each dataset, test: branch preprocessing hashes and unchanged parent files;
new backend instance/session navigation; lost-response replay with stable key;
rejection of incompatible time/random scores; refusal to finalize another model
on an already used final holdout; refusal to export a deliberately mutated model
(restore bytes afterward); standalone exported retraining using an independent
wheel-installed interpreter. Direct MLflow with explicit metadata/hash/key checks
is a positive control: equivalent safeguards can be programmed there too.

Count only predefined checks, report failures rather than silently change the
denominator. These are injected design cases, not real-world failure rates.
Timing is descriptive: serial execution/cache/order effects prohibit speed claims.

## DSH-specific integration

Run a real DSH PTC workflow on the prepared retail CSV under workspace-write
policy: train, branch with fitted preprocessing, compare, finalize, export; replay
stable request keys; dispose/remount the plugin and recover runs by Session.
Record outer/nested calls and sandbox policy. No LLM is involved in this integration
test; it cannot establish fewer model tokens, fewer human interventions, or faster
natural-language Agent work. Existing live pilot limitations remain in BENCHMARK.md.

Any discovered defect and protocol deviation must be recorded in the report.
