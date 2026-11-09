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
        ExperimentSpec,
        ExportRequest,
        ListRequest,
        RunRequest,
        WorkbenchError,
    )

    args = dict(request.args)
    operation = request.operation
    if operation == "health":
        from . import __version__
        from .files import environment
        if args:
            raise WorkbenchError("INVALID_ARGUMENTS", "health takes no arguments")
        return {"version": __version__, "protocol_version": 1, "tracking_uri": workbench.tracking_uri,
                "root": str(workbench.root), "allow_custom_code": workbench.allow_custom_code,
                "environment": environment()}
    if operation == "create":
        args = CreateRequest.model_validate(args).model_dump()
        request_key = args.pop("request_key", None)
        return workbench.create(ExperimentSpec.model_validate(args), request_key)
    schemas = {"start": RunRequest, "get": RunRequest, "list": ListRequest, "cancel": RunRequest,
               "branch": BranchRequest, "compare": CompareRequest, "export": ExportRequest}
    validated = schemas[operation].model_validate(args).model_dump()
    return getattr(workbench, operation)(**validated)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["rpc", "worker"])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--allow-custom-code", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--token")
    options = parser.parse_args()
    if options.command == "worker":
        from .worker import run_worker
        return run_worker(options.root, options.run_id, options.token)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from .service import Workbench
            from .spec import RpcRequest, WorkbenchError
            raw = sys.stdin.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise WorkbenchError("REQUEST_TOO_LARGE", "RPC input exceeds 1 MiB")
            request = RpcRequest.model_validate_json(raw)
            workbench = Workbench(options.root, allow_custom_code=options.allow_custom_code)
            result = dispatch(workbench, request)
        response = {"protocol_version": 1, "ok": True, "result": result}
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
