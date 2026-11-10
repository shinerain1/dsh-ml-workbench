"""One JSON RPC request per process; the worker has an independent lifetime."""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["MLFLOW_ENABLE_TELEMETRY"] = "false"


def dispatch(workbench, request):
    from .spec import (
        BranchRequest,
        CompareRequest,
        CreateRequest,
        DiffRequest,
        ExperimentSpec,
        ExportRequest,
        InspectRequest,
        ListRequest,
        PredictRequest,
        ResumeRequest,
        RunRequest,
        SearchCreateRequest,
        SearchRequest,
        WaitRequest,
        WorkbenchError,
    )

    args = dict(request.args)
    from . import __version__
    if request.expected_backend_version is not None and request.expected_backend_version != __version__:
        raise WorkbenchError("VERSION_MISMATCH", f"Plugin expects {request.expected_backend_version}, backend is {__version__}; install matching release wheel")
    operation = request.operation
    if operation in {"create", "run", "branch"} and args.get("request_key") is None and workbench.dsh_context:
        from .files import digest
        args["request_key"] = "dsh:" + digest({"operation": operation, **workbench.dsh_context})
    if operation == "health":
        from importlib.metadata import version

        from .files import environment
        if args:
            raise WorkbenchError("INVALID_ARGUMENTS", "health takes no arguments")
        return {"version": __version__, "protocol_version": 1, "tracking_uri": workbench.tracking_uri,
                "root": str(workbench.root), "allow_custom_code": workbench.allow_custom_code,
                "environment": environment(), "optuna_version": version("optuna"), "python_executable": sys.executable,
                "isolated_environment": sys.prefix != sys.base_prefix,
                "features": ["random/group/time splits", "Optuna ask/tell", "batch prediction", "experiment diff", "HTML export",
                             "DSH Session recovery", "DSH invocation replay identity", "independent final test",
                             "background training deadline", "bounded wait", "external confined workers"]}
    if operation in {"create", "run"}:
        args = CreateRequest.model_validate(args).model_dump()
        request_key = args.pop("request_key", None)
        result = workbench.create(ExperimentSpec.model_validate(args), request_key)
        return workbench.start(result["run_id"]) if operation == "run" else result
    schemas = {"resume": ResumeRequest, "wait": WaitRequest, "finalize": RunRequest,
               "start": RunRequest, "get": RunRequest, "list": ListRequest, "cancel": RunRequest,
               "branch": BranchRequest, "compare": CompareRequest, "export": ExportRequest,
               "inspect": InspectRequest, "diff": DiffRequest, "predict": PredictRequest,
               "search_create": SearchCreateRequest, "search_step": SearchRequest,
               "search_get": SearchRequest, "search_cancel": SearchRequest}
    validated = schemas[operation].model_validate(args).model_dump()
    result = getattr(workbench, operation)(**validated)
    if result.get("run_id"):
        workbench.store.link(workbench.dsh_context, "run", result["run_id"])
    if result.get("study_name"):
        workbench.store.link(workbench.dsh_context, "study", result["study_name"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["rpc", "worker", "supervise", "doctor"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--allow-custom-code", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--token")
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--parent-created", type=float)
    parser.add_argument("--external-workers", action="store_true")
    if "doctor" in sys.argv[1:2]:
        from .doctor import diagnose
        options = parser.parse_args()
        report = diagnose(options.root)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["ok"] else 1
    options = parser.parse_args()
    if options.command == "supervise":
        from .supervisor import supervise
        return supervise(options.root, options.run_id, options.token)
    if options.command == "worker":
        from .worker import run_worker
        return run_worker(options.root, options.run_id, options.token, options.parent_pid, options.parent_created)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from .service import Workbench
            from .spec import RpcRequest, WorkbenchError
            raw = sys.stdin.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise WorkbenchError("REQUEST_TOO_LARGE", "RPC input exceeds 1 MiB")
            request = RpcRequest.model_validate_json(raw)
            workbench = Workbench(options.root, allow_custom_code=options.allow_custom_code,
                                  dsh_context=request.dsh_context.model_dump() if request.dsh_context else None,
                                  external_workers=options.external_workers)
            result = dispatch(workbench, request)
        response = {"protocol_version": 1, "ok": True, "result": result, "launches": workbench.pending_launches}
    except Exception as error:
        code = getattr(error, "code", "BACKEND_ERROR")
        if isinstance(error, (TypeError, ValueError)):
            code = "INVALID_ARGUMENTS"
        response = {"protocol_version": 1, "ok": False,
                    "error": {"code": code, "message": str(error)[:2000]}}
    print(json.dumps(response, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
