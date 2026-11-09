import pytest
from dsh_ml_workbench.__main__ import dispatch
from dsh_ml_workbench.spec import ExperimentSpec, RpcRequest
from pydantic import ValidationError


@pytest.mark.parametrize("request_key", [[], 3, "", "x" * 257])
def test_create_rejects_malformed_idempotency_key_before_side_effects(request_key):
    request = RpcRequest(operation="create", args={"dataset_path": "unused.csv", "target": "y",
                         "task": "classification", "request_key": request_key})
    with pytest.raises(ValidationError):
        dispatch(None, request)


def test_native_sklearn_controls_are_normalized_without_changing_caller_input():
    args = {"dataset_path": "data.csv", "target": "target", "task": "classification", "seed": 19,
            "params": {"random_state": 19, "n_jobs": 1, "n_estimators": 12}}
    normalized = ExperimentSpec.model_validate(args)
    assert normalized.params == {"n_estimators": 12} and normalized.seed == 19
    assert args["params"] == {"random_state": 19, "n_jobs": 1, "n_estimators": 12}


@pytest.mark.parametrize("params", [{"random_state": 7}, {"random_state": None}, {"n_jobs": -1},
                                    {"n_jobs": True}, {"random_state": 42.0}])
def test_native_controls_cannot_override_managed_execution_policy(params):
    with pytest.raises(ValidationError, match="conflicting execution controls"):
        ExperimentSpec.model_validate({"dataset_path": "data.csv", "target": "target", "task": "classification",
                                       "params": params})
