"""Publish sanitized evidence from actual pytest JUnit results (no host or path metadata)."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from dsh_ml_workbench import __version__
from dsh_ml_workbench.files import atomic_json, environment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--xml", action="append", type=Path, required=True)
args = parser.parse_args()
cases = []
for path in args.xml:
    tree = ET.parse(path)
    for case in tree.findall(".//testcase"):
        assert case.find("failure") is None and case.find("error") is None and case.find("skipped") is None, case.attrib
        cases.append({"name": case.attrib["classname"] + "." + case.attrib["name"], "seconds": float(case.attrib["time"])})
assert len({case["name"] for case in cases}) == len(cases), "Duplicate cases would inflate the count"
atomic_json(Path(__file__).resolve().parents[1] / "docs" / "test-results-v2.json",
            {"version": __version__, "environment": environment(), "passed": len(cases), "cases": cases})
print({"passed": len(cases), "version": __version__})
