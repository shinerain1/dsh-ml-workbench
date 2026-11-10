"""Generate backend request schemas and the tested environment lock without local paths."""

import json
from pathlib import Path

from dsh_ml_workbench.files import requirements
from dsh_ml_workbench.spec import (
    BranchRequest,
    CompareRequest,
    CreateRequest,
    DiffRequest,
    ExportRequest,
    InspectRequest,
    ListRequest,
    PredictRequest,
    ResumeRequest,
    RpcRequest,
    RunRequest,
    SearchCreateRequest,
    SearchRequest,
    WaitRequest,
)

root = Path(__file__).resolve().parents[1]
create = CreateRequest.model_json_schema()
schemas = {"rpc": RpcRequest.model_json_schema(), "create": create, "run": create}
for operation, model in {"resume": ResumeRequest, "start": RunRequest, "wait": WaitRequest, "finalize": RunRequest,
                         "get": RunRequest, "cancel": RunRequest,
                         "list": ListRequest, "branch": BranchRequest, "compare": CompareRequest,
                         "export": ExportRequest, "inspect": InspectRequest, "diff": DiffRequest,
                         "predict": PredictRequest, "search_create": SearchCreateRequest,
                         "search_step": SearchRequest, "search_get": SearchRequest, "search_cancel": SearchRequest}.items():
    schemas[operation] = model.model_json_schema()
schemas["health"] = {"type": "object", "additionalProperties": False, "properties": {}}
(root / "docs" / "api-schema.json").write_text(json.dumps({"protocol_version": 1, "schemas": schemas},
    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(root / "requirements.lock").write_text(requirements(), encoding="utf-8")
