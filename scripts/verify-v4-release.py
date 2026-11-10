"""Validate installed-wheel imports, final evaluation and standalone retraining in a clean venv."""

import argparse
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    python = str(args.python.resolve())
    project = Path(__file__).resolve().parents[1]
    folder = Path(tempfile.mkdtemp(prefix="v4-package-", dir=project / ".verification"))
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONPATH": ""}

    def execute(argv, cwd=folder, input=None):
        completed = subprocess.run([python, *argv], cwd=cwd, input=input, text=True, encoding="utf-8",
                                   capture_output=True, timeout=180, env=env)
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        return json.loads(completed.stdout.strip().splitlines()[-1])

    imported = execute(["-c", "import json,dsh_ml_workbench; print(json.dumps({'version':dsh_ml_workbench.__version__,'path':dsh_ml_workbench.__file__}))"])
    assert imported["version"] == "0.4.0" and "site-packages" in imported["path"], imported
    doctor = execute(["-m", "dsh_ml_workbench", "doctor", "--root", str(folder / "doctor")])
    assert doctor["ok"] and doctor["isolated_environment"], doctor

    def reproduce(archive, name):
        destination = folder / name
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(destination)
        result = execute(["reproduce.py"], cwd=destination)
        assert result["metrics_match"] and result["final_test_metrics_match"], result
        return result

    local = json.loads((project / ".verification/dsh-ptc-v4-local.json").read_text(encoding="utf-8"))
    classification = reproduce(local["completed"]["result"]["delivery"], "classification")
    root = folder / "records"
    def rpc(operation, args):
        response = execute(["-m", "dsh_ml_workbench", "rpc", "--root", str(root)],
            input=json.dumps({"protocol_version": 1, "operation": operation, "args": args,
                              "expected_backend_version": "0.4.0"}))
        assert response["ok"], response
        return response["result"]
    data = folder / "regression.csv"
    data.write_text("x,z,target\n" + "\n".join(f"{i/7},{i%3},{2*i/7+(i%3)/2}" for i in range(100)), encoding="utf-8")
    run = rpc("run", {"dataset_path": str(data), "target": "target", "task": "regression", "model": "ridge", "request_key": "wheel-regression"})
    for _ in range(4):
        run = rpc("wait", {"run_id": run["run_id"], "wait_seconds": 30})
        if run["state"] not in {"starting", "running"}:
            break
    assert run["state"] == "succeeded", run
    final = rpc("finalize", {"run_id": run["run_id"]})
    assert "rmse" in final["final_test"]["metrics"]
    delivery = rpc("export", {"run_id": run["run_id"], "include_data": True})
    regression = reproduce(delivery["path"], "regression")
    report = {"version": imported["version"], "installed_wheel_import": True,
              "isolated_virtualenv": doctor["isolated_environment"], "doctor_checks": doctor["checks"],
              "classification_retraining": classification, "regression_retraining": regression,
              "validation_and_final_test_reproduced": True}
    (project / "docs/packaging-v4-results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
