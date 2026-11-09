# Version 0.3.0 validation

This release focuses on DSH-native behavior and evidence-driven fixes. [The comparison report](BENCHMARK.md) includes successful safeguards and unfavorable findings; it does not claim upstream parity or general Agent productivity gains.

## Executed checks

- **31 distinct Python cases** in [test-results-v3.json](test-results-v3.json). The full 25-case suite passed in 279.25 seconds. After the pilot exposed parameter friction, 12 focused protocol/Session cases passed in 12.11 seconds, including six new cases. Repeated names are replaced with their latest passing result rather than counted again. One upstream SQLAlchemy deprecation warning remains.
- **4 distinct Node cases**: three registry/lifecycle/error cases passed in 158.77 seconds; a real delayed Python control-process test passed in 2.17 seconds, confirming timeout reporting preserves uncertainty about committed work and releases the process. An earlier concurrent cold-start run exceeded its 120-second control timeout; the successful rerun retained the same timeout. This remains a startup limitation.
- TypeScript build/build-check, Python lint and installed-environment dependency checks passed.
- Actual DSH checkout `8bd073392eef6f54e921a093b95076248eb08cbb` (`0.2.2-alpha.3`) exercised Session identity, plugin reconstruction, exact replay, cross-Session separation, guard denial and pre-aborted calls through its Native executor. [Result](dsh-session-results.json). There are 17 tools.
- Four paired direct-MLflow/plugin experiments produced identical metrics, complete records and successful reproductions. Three selected fault cases were checked against raw MLflow and explicit-check positive controls. [Result](benchmark-results.json).
- One live two-turn trial per arm completed using `deepseek-official/deepseek-flash`, low reasoning effort and a stored API credential. Turn two adopted the persisted Session in a new process. Independent artifact/partition/preprocessing/reproduction auditing is in [agent-pilot-results.json](agent-pilot-results.json). DSH's sandbox remained enabled in the eligible temporary workspace.
- Both failed model-generated creation payloads succeeded after the fix, preserving original run IDs and count. [Result](pilot-regression-results.json). This replay is not represented as a new live trial.

## Packaging and limits

The wheel is installed into the previously isolated release environment with unchanged locked numerical dependencies, not an editable import. Native DSH checks also run against that installed backend. Independent pilot delivery audits use the release interpreter. Historical v0.2/v0.1 reports are retained without relabeling old tests.

The public v0.3.0 Git commit was installed through DSH's `plugin --profile headless add` into an isolated profile. That installed plugin registered 17 tools, executed health through the actual Native executor with the installed wheel, and unloaded cleanly. The public index returned HTTP 200, passed DSH's own index parser and retained v0.2.0/v0.1.0. All five GitHub release asset digests match local SHA-256 values in [publication-v3.json](publication-v3.json). Installation emitted peer-dependency warnings; the assembled runtime check passed.

The earlier account-provider sign-in failure was avoided by using the already configured API-key provider. No credential is copied into the repository. Raw model transcripts, actual Session IDs, local databases and generated training outputs remain outside Git; reports contain aggregate measurements and sanitized results.

Limits include single-user local execution, startup/poll overhead, no OS sandbox around Python, no weight checkpoint resume, no distributed scheduling/nested CV, and no established cross-platform reliability. PTC-facing schemas/example are provided, but the live pilot used Native mode and is not a PTC performance measurement.
