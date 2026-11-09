"""Run one prepared pilot turn through the supported DSH profile launcher; keep raw logs local."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("arm", choices=["with-plugin", "without-plugin"])
parser.add_argument("--turn", type=int, choices=[1, 2], default=1)
parser.add_argument("--name", default="agent-pilot-v3")
parser.add_argument("--root", type=Path)
parser.add_argument("--checkout", type=Path, default=Path("E:/dsharness"))
args = parser.parse_args()
project = Path(__file__).resolve().parents[1]
if Path(args.name).name != args.name:
    raise ValueError("name must be one directory name")
folder = (args.root.resolve() if args.root else project / ".verification" / args.name) / args.arm
environment = {**os.environ, "DSH_HOME": str(folder / "dsh-home"),
               "TSX_TSCONFIG_PATH": str(args.checkout / "tsconfig.json"), "NODE_USE_ENV_PROXY": "1"}
command = ["node", "--import", (args.checkout / "node_modules/tsx/dist/esm/index.mjs").as_uri(),
           str(args.checkout / "apps/cli/src/bin.ts"), "headless", "--patch", str(folder / "overlay.yml"),
           "--approval-policy", "never", "--json"]
if args.turn == 2:
    events = [json.loads(line) for line in (folder / "turn-1.jsonl").read_text(encoding="utf-8").splitlines() if line.startswith("{")]
    session = next(e["sessionId"] for e in events if e["type"] == "session")
    command += ["--session-id", session]
command += ["-"]
task = (folder / ("task.txt" if args.turn == 1 else "followup.txt")).read_text(encoding="utf-8")
started = time.monotonic()
with (folder / f"turn-{args.turn}.jsonl").open("w", encoding="utf-8") as out, (folder / f"turn-{args.turn}.stderr").open("w", encoding="utf-8") as err:
    try:
        result = subprocess.run(command, input=task, text=True, encoding="utf-8", cwd=folder, env=environment,
                                stdout=out, stderr=err, timeout=1200)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = "deadline"
print(json.dumps({"arm": args.arm, "turn": args.turn, "exit_code": code, "seconds": round(time.monotonic() - started, 2)}))
