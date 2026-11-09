import pytest
from dsh_ml_workbench.__main__ import dispatch
from dsh_ml_workbench.service import Workbench
from dsh_ml_workbench.spec import RpcRequest, WorkbenchError
from pydantic import ValidationError


def test_dsh_replay_resume_and_cross_session_navigation(tmp_path, classification_csv):
    root = tmp_path / "records"
    context = {"session_id": "session-a", "call_id": "create-1"}
    workbench = Workbench(root, dsh_context=context)
    args = {"dataset_path": str(classification_csv), "target": "target", "task": "classification"}
    request = RpcRequest(operation="create", args=args)
    first = dispatch(workbench, request)
    # The result was lost: reconstruct the control process and replay the same DSH call.
    reopened = Workbench(root, dsh_context=context)
    assert dispatch(reopened, request)["run_id"] == first["run_id"]
    assert len(reopened.list()["runs"]) == 1
    assert reopened.client.get_run(first["run_id"]).data.tags["mlwb.dsh.session_id"] == "session-a"
    with pytest.raises(WorkbenchError, match="different arguments"):
        dispatch(reopened, RpcRequest(operation="create", args={**args, "seed": 13}))
    # A genuinely new call remains a new experiment, even with the same parameters.
    other = Workbench(root, dsh_context={"session_id": "session-b", "call_id": "create-1"})
    second = dispatch(other, request)
    assert second["run_id"] != first["run_id"]
    own = other.resume()
    assert [r["ref"] for r in own["resources"]] == [second["run_id"]]
    assert [r["ref"] for r in other.resume("session-a")["resources"]] == [first["run_id"]]
    # Existing explicit idempotency keys still work across different Session identities.
    explicit = RpcRequest(operation="create", args={**args, "request_key": "intent-1"})
    one = dispatch(workbench, explicit)
    assert dispatch(other, explicit)["run_id"] == one["run_id"]
    page = other.resume(limit=1)
    assert page["next_offset"] == 1
    assert len(other.resume(offset=1)["resources"]) == 1
    study = dispatch(workbench, RpcRequest(operation="search_create", args={
        "study_name": "recover-study", "run_id": first["run_id"], "search_space": {
            "max_depth": {"type": "int", "low": 2, "high": 3}}, "n_trials": 1}))
    assert study["trials"] == []
    recovered = Workbench(root, dsh_context=context).resume()
    reference = next(r for r in recovered["resources"] if r["kind"] == "study")
    assert reference["next_tool"] == "ml_search_get" and reference["state"] == "unverified"
    assert workbench.search_get("recover-study")["trials"] == []
    assert all(r["state"] == "created" for r in workbench.list()["runs"])
    branch_call = RpcRequest(operation="branch", args={"run_id": first["run_id"], "reuse_preprocessing": False,
                                                       "changes": {"name": "continued"}})
    child = dispatch(workbench, branch_call)
    assert child["parent_run_id"] == first["run_id"]
    assert dispatch(Workbench(root, dsh_context=context), branch_call)["run_id"] == child["run_id"]
    assert any(r["ref"] == child["run_id"] for r in workbench.resume()["resources"])
    # Automatic Session metadata is not an accepted model-generated argument.
    with pytest.raises(ValidationError):
        dispatch(workbench, RpcRequest(operation="create", args={**args, "dsh_context": context}))
    with pytest.raises(WorkbenchError, match="DSH Session"):
        Workbench(root).resume()


def test_study_intent_is_discoverable_if_optuna_creation_fails(tmp_path, regression_csv, monkeypatch):
    import optuna
    workbench = Workbench(tmp_path / "records", dsh_context={"session_id": "study-owner", "call_id": "1"})
    base = dispatch(workbench, RpcRequest(operation="create", args={
        "dataset_path": str(regression_csv), "target": "target", "task": "regression", "model": "ridge"}))
    def fail(**kwargs):
        raise RuntimeError("injected storage outage")
    monkeypatch.setattr(optuna, "create_study", fail)
    with pytest.raises(RuntimeError, match="storage outage"):
        workbench.search_create(study_name="pending-intent", run_id=base["run_id"],
                               search_space={"alpha": {"type": "float", "low": 0.1, "high": 1.0}})
    resources = Workbench(workbench.root).resume("study-owner")["resources"]
    assert any(r["ref"] == "pending-intent" and r["state"] == "unverified" for r in resources)
