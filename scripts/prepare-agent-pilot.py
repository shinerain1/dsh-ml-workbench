"""Prepare identical synthetic tasks for a small live DSH pilot, without copying secrets."""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import make_classification

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--name", default="agent-pilot-v3")
parser.add_argument("--root", type=Path)
parser.add_argument("--strict-retrain", action="store_true", help="Strengthen future pilots: require estimator refitting, not just rescoring")
args = parser.parse_args()
if Path(args.name).name != args.name:
    raise ValueError("name must be one directory name")
project = Path(__file__).resolve().parents[1]
root = args.root.resolve() if args.root else project / ".verification" / args.name
root.mkdir(exist_ok=False)
x, y = make_classification(n_samples=160, n_features=6, n_informative=4, random_state=7)
frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(6)])
frame["target"] = y
python = str(project / ".venv" / "Scripts" / "python.exe")
common = f"""This is a bounded experiment workflow evaluation on synthetic data. Work only in the current directory. Do not delegate to subagents, access the network, install packages, or modify other projects. Your Python interpreter is {python!r}; sklearn, pandas, MLflow and Optuna are installed. Use existing ml_* tools if available; otherwise use Python and MLflow directly. Do not import dsh_ml_workbench in Python. No user clarification is needed. Keep calls and output concise.

Train a binary classification experiment from customers.csv (target column target), then a second experiment on EXACTLY the same stratified holdout (test_fraction 0.2, random seed 42). First model: RandomForestClassifier(n_estimators=12, random_state=42, n_jobs=1). Second: ExtraTreesClassifier(n_estimators=12, random_state=42, n_jobs=1). Fit median imputation and StandardScaler only on training rows. Record the code, original data, parameters, exact split, fitted preprocessing, metrics and model artifacts in MLflow or the workbench. Compare accuracy only if evaluation conditions match. Preserve both experiments for the next turn. Do not export yet. Write pilot-result.json with an experiments array (run_id, model, accuracy, location), and say which run is the second. Complete the work now, using no more than 18 tool calls if possible."""
followup = """Continue the previous task after this real DSH process restart. Keep the second experiment's EXACT fitted preprocessing (copy/reuse it without fitting), data and validation rows. Create a NEW LogisticRegression(max_iter=1000, random_state=42) experiment, compare accuracy with the first two, and leave both original models unchanged. Export the new experiment (not necessarily the winner) with model, code, CSV, environment and a standalone reproduction script. Actually run its reproduction and verify the recorded metrics. Update pilot-result.json with the new run, parent relation, reused-preprocessor hash equality, delivery path and reproduction result. Work only here; no delegation/network/install. Use the same interpreter. Do not import dsh_ml_workbench in Python. Finish without clarification."""
if args.strict_retrain:
    followup += " The standalone script MUST instantiate and fit a NEW estimator on the frozen training rows using the copied fitted preprocessor, then compare all recorded validation metrics. Merely loading the saved estimator and rescoring it does not satisfy this task. It must work from a copied delivery directory without the original MLflow database or workbench."
for arm in ("without-plugin", "with-plugin"):
    folder = root / arm
    folder.mkdir()
    frame.to_csv(folder / "customers.csv", index=False)
    (folder / "task.txt").write_text(common, encoding="utf-8")
    (folder / "followup.txt").write_text(followup, encoding="utf-8")
    rows = [
        {"id": "credentials", "disabled": False, "config": {"path": str(Path.home() / ".dsh/.credentials.yaml"), "watch": False}},
        {"id": "agent-default-model", "config": {"provider": "deepseek-official", "model": "deepseek-flash", "reasoningEffort": "low"}},
        {"id": "llm-deepseek", "config": {"maxTokens": 8192, "retryPolicy": {"mode": "normal", "maxRetries": 0}, "streamIdleTimeoutMs": 60000}},
    ]
    if arm == "with-plugin":
        rows.append({"insert": [{"id": "ml-workbench", "name": str(project / "lib/index.js"),
                                 "config": {"pythonExecutable": python, "root": str(folder / "records"),
                                            "requestTimeoutMs": 300000}}]})
    (folder / "overlay.yml").write_text(yaml.safe_dump(rows, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(json.dumps({"root": str(root), "arms": 2, "turns_per_arm": 2, "model": "deepseek-flash", "reasoning": "low", "strict_retrain": args.strict_retrain}))
