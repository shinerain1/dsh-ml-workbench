Adds `shinerain1/dsh-ml-workbench` under `workflow` as one YAML entry.

DSH ML Workbench adapts MLflow, Optuna and scikit-learn into DSH tools for local CSV classification and regression experiments. It records immutable inputs and artifacts, branches experiments with fitted preprocessing reuse, checks evaluation compatibility, and exports a standalone retraining package. It requires a separately installed Python backend; installing the DSH bundle alone does not install Python or its dependencies.

- The repository declares `dsh.bundle` with `cordis.patch.yml` and has the `dsh-plugin` topic.
- The entry changes only `data/plugins/shinerain1__dsh-ml-workbench.yml`; generated READMEs and other entries are untouched.
- Bilingual descriptions state supported behavior and the backend requirement.
- `tarball` pins the prebuilt v0.4.0 GitHub Release asset; `icon` is a GitHub-hosted PNG.
- DSH host runtime dependencies (`cordis` and `dsh-tools`) are peers. Training and tracking reuse the upstream Python libraries.

Installation and scope: https://github.com/shinerain1/dsh-ml-workbench/blob/main/docs/MARKETPLACE.md

Reviewable evidence:

- Claim-to-code/test map: https://github.com/shinerain1/dsh-ml-workbench/blob/main/docs/EVIDENCE.md
- Release validation, including installed-wheel and real DSH host loading: https://github.com/shinerain1/dsh-ml-workbench/blob/main/docs/VALIDATION-v0.4.md
- UCI retail classification and bike regression: https://github.com/shinerain1/dsh-ml-workbench/blob/main/docs/REAL-DATA-EXPERIMENTS.md
- Machine-readable results: https://github.com/shinerain1/dsh-ml-workbench/blob/main/docs/real-data-results.json
- Actual DSH PTC / Session / sandbox integration: https://github.com/shinerain1/dsh-ml-workbench/blob/main/docs/dsh-real-data-results.json

The 18 paired real-data experiments matched direct MLflow's metrics. Twelve branches preserved fitted preprocessing and parent outputs; fourteen predefined workflow checks passed; both exported selected models were independently retrained. These are controlled workflow checks, not evidence of improved Agent productivity or production incident rates. The report includes process overhead, limitations, and the earlier unfavorable live-Agent pilot.
