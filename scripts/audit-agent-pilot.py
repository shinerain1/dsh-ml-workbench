"""Independently audit completed live pilot outputs and emit path-free measurements."""
import argparse
import collections
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from dsh_ml_workbench.files import read_json, sha256
from dsh_ml_workbench.service import Workbench

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--python", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("docs/agent-pilot-results.json"))
args = parser.parse_args()
root = args.root.resolve()
project = Path(__file__).resolve().parents[1]
audit = Path(tempfile.mkdtemp(prefix="pilot-audit-", dir=project / ".verification"))
hook = audit / "hook"
hook.mkdir()
(hook / "sitecustomize.py").write_text('''import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
original = LogisticRegression.fit
def counted(self, *args, **kwargs):
    with Path(os.environ['MLWB_AUDIT_COUNTER']).open('a') as stream: stream.write('fit\\n')
    return original(self, *args, **kwargs)
LogisticRegression.fit = counted
''', encoding="utf-8")
report = {"kind": "live-dsh-agent-pilot", "replicates_per_arm": 1, "turns_per_arm": 2,
          "model": "deepseek-official/deepseek-flash", "reasoning_effort": "low", "arms": {},
          "excluded_setup_attempts": 2, "excluded_reason": "DSH Windows sandbox ACL failure in the E-drive workspace",
          "posthoc_measurement": "Estimator fit-call instrumentation was added after noticing that baseline reproduction only reloads a model; it was not a preregistered success gate."}
for arm in ("without-plugin", "with-plugin"):
    folder = root / arm
    pilot = read_json(folder / "pilot-result.json")
    before = read_json(folder / "before-model-hashes.json")
    assert before and all(sha256(folder / path) == digest for path, digest in before.items()), arm
    events = [[json.loads(line) for line in (folder / f"turn-{i}.jsonl").read_text(encoding="utf-8").splitlines()] for i in (1, 2)]
    sessions = [next(e["sessionId"] for e in turn if e["type"] == "session") for turn in events]
    assert sessions[0] == sessions[1]
    assert all(any(e["type"] == "final" and e["reason"]["kind"] == "completed" for e in turn) for turn in events)
    usage = collections.Counter()
    calls = collections.Counter()
    errors = []
    for turn in events:
        for event in turn:
            if event["type"] == "tool_call":
                calls[event["tool"]] += 1
            if event["type"] == "tool_result" and event["status"] == "error":
                errors.append(event["result"].splitlines()[0])
            if event["type"] == "status" and event.get("phase") == "step_end":
                usage.update(event.get("usage", {}))
    delivery = audit / arm
    if arm == "without-plugin":
        source = Path(pilot["delivery"]["path"])
        shutil.copytree(source, delivery)
        parent_pre = folder / "mlflow-artifacts" / pilot["second_run_id"] / "artifacts/preprocessing/preprocessor.joblib"
        preprocessor_equal = sha256(parent_pre) == sha256(delivery / "preprocessor.joblib")
        plain_split = read_json(delivery / "split.json")
        split = {"train": plain_split["train_index"], "test": plain_split["test_index"]}
        parent_link = pilot["parent_run_id"] == pilot["second_run_id"]
    else:
        with zipfile.ZipFile(folder / pilot["delivery_path"]) as archive:
            archive.extractall(delivery)
        workbench = Workbench(folder / "records")
        record = workbench.get(pilot["new_run_id"])
        parent = workbench.run_dir(pilot["second_run_id"])
        preprocessor_equal = sha256(parent / "output/preprocessor.pkl") == sha256(delivery / "inputs/reused_preprocessor.pkl")
        split = read_json(delivery / "inputs/split.json")
        parent_link = record["parent_run_id"] == pilot["second_run_id"]
        assert record["metrics"]["accuracy"] == 0.84375
    assert preprocessor_equal and parent_link
    if arm == "without-plugin":
        reference_split = split
    else:
        assert all(set(split[part]) == set(reference_split[part]) for part in ("train", "test"))
        report["cross_arm_same_partition_rows"] = True
        report["cross_arm_row_order_equal"] = split == reference_split
    count_file = audit / f"{arm}-fits.txt"
    environment = {**os.environ, "PYTHONPATH": str(hook), "MLWB_AUDIT_COUNTER": str(count_file),
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MLFLOW_ENABLE_TELEMETRY": "false"}
    command = [str(args.python.resolve()), "reproduce.py"]
    if arm == "with-plugin":
        command += ["--output", str(audit / "freshly-trained")]
    outcome = subprocess.run(command, cwd=delivery, env=environment, capture_output=True, text=True, timeout=180)
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    fit_calls = len(count_file.read_text().splitlines()) if count_file.exists() else 0
    if arm == "with-plugin":
        assert json.loads(outcome.stdout.splitlines()[-1])["metrics_match"]
    else:
        assert read_json(delivery / "reproduction_result.json")["accuracy_match"]
    assert [e["accuracy"] for e in pilot["experiments"]] == [0.8125, 0.84375, 0.84375]
    report["arms"][arm] = {"accuracies": [e["accuracy"] for e in pilot["experiments"]],
        "same_session_restored": True, "old_artifact_hashes_unchanged": len(before),
        "parent_link_verified": parent_link, "fitted_preprocessor_bytes_equal": preprocessor_equal,
        "isolated_delivery_metric_reproduction": True, "estimator_fit_calls_during_reproduction": fit_calls,
        "tool_calls_by_turn": [sum(e["type"] == "tool_call" for e in turn) for turn in events],
        "tool_calls": dict(calls), "executor_reported_tool_errors": len(errors), "error_summaries": errors,
        "usage": dict(usage), "extra_user_answers": 0}
assert report["arms"]["without-plugin"]["estimator_fit_calls_during_reproduction"] == 0
assert report["arms"]["with-plugin"]["estimator_fit_calls_during_reproduction"] == 1
args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report))
