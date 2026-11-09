import pytest
from dsh_ml_workbench.__main__ import dispatch
from dsh_ml_workbench.spec import RpcRequest
from pydantic import ValidationError


@pytest.mark.parametrize("request_key", [[], 3, "", "x" * 257])
def test_create_rejects_malformed_idempotency_key_before_side_effects(request_key):
    request = RpcRequest(operation="create", args={"dataset_path": "unused.csv", "target": "y",
                         "task": "classification", "request_key": request_key})
    with pytest.raises(ValidationError):
        dispatch(None, request)
