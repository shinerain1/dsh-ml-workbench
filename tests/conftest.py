import os
import time

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

os.environ["MLFLOW_ENABLE_TELEMETRY"] = "false"


@pytest.fixture
def classification_csv(tmp_path):
    x, y = make_classification(n_samples=160, n_features=6, n_informative=4, random_state=7)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    frame["channel"] = ["web" if i % 2 else "shop" for i in range(len(frame))]
    frame["target"] = y
    path = tmp_path / "customers.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def regression_csv(tmp_path):
    x, y = make_regression(n_samples=140, n_features=5, noise=2, random_state=9)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
    frame["target"] = y
    path = tmp_path / "regression.csv"
    frame.to_csv(path, index=False)
    return path


def wait_run(workbench, run_id, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = workbench.get(run_id)
        if result["state"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            return result
        time.sleep(0.1)
    workbench.cancel(run_id)
    raise AssertionError(f"Run {run_id} did not settle within {timeout}s")


def wait_log(workbench, run_id, marker, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = workbench.get(run_id)
        if marker in result.get("log_tail", ""):
            return result
        if result["state"] in {"failed", "interrupted"}:
            raise AssertionError(result)
        time.sleep(0.1)
    workbench.cancel(run_id)
    raise AssertionError(f"Worker did not reach {marker}")
