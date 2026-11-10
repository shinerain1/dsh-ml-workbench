"""Publish sanitized evidence from actual pytest JUnit results (no host or path metadata)."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from dsh_ml_workbench import __version__
from dsh_ml_workbench.files import atomic_json, environment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--xml", action="append", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("docs/test-results-v2.json"))
parser.add_argument("--latest-wins", action="store_true", help="Keep the latest passing result of each repeated case")
args = parser.parse_args()
cases = []
for path in args.xml:
    tree = ET.parse(path)
    for case in tree.findall(".//testcase"):
        passed = all(case.find(kind) is None for kind in ("failure", "error", "skipped"))
        cases.append({"name": case.attrib["classname"] + "." + case.attrib["name"], "seconds": float(case.attrib["time"]), "passed": passed})
if args.latest_wins:
    cases = list({case["name"]: case for case in cases}.values())
assert len({case["name"] for case in cases}) == len(cases), "Duplicate cases would inflate the count"
assert all(case.pop("passed") for case in cases), "Every final case must pass; rerun failures before recording validation"
atomic_json(args.output,
            {"version": __version__, "environment": environment(), "passed": len(cases), "cases": cases})
print({"passed": len(cases), "version": __version__})
