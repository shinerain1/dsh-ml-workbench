"""Create a local-only DSH overlay without copying or printing credentials."""
import json
import os
from pathlib import Path

import yaml

root = Path.cwd().resolve()
local = root / ".verification"
local.mkdir(exist_ok=True)
credentials = Path.home() / ".dsh" / ".credentials.yaml"
rows = [
    {"id": "credentials", "disabled": False, "config": {"path": str(credentials), "watch": False}},
    {"id": "authorization", "disabled": False},
    {"id": "deepseek-account", "disabled": False},
    {"id": "agent-default-model", "config": {"provider": "deepseek-account", "model": "deepseek-flash"}},
    {"id": "llm-deepseek-account", "config": {"maxTokens": 8192, "retryPolicy": {"mode": "normal", "maxRetries": 0}}},
    {"insert": [{"id": "ml-workbench", "name": str(root / "lib" / "index.js"), "config": {
        "pythonExecutable": str(root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
        "root": str(local / "live-records"), "allowCustomCode": True,
    }}]},
]
path = local / "live.patch.yml"
path.write_text(yaml.safe_dump(rows, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(json.dumps({"patch": str(path), "credentials_available": credentials.is_file()}))
