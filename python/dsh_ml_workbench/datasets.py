"""CSV diagnostics and explicit evaluation boundaries, using sklearn splitters."""

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .files import sha256
from .spec import ExperimentSpec, WorkbenchError

MAX_DATASET_BYTES = 128 * 1024 * 1024


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size > MAX_DATASET_BYTES:
        raise WorkbenchError("INVALID_DATASET", "CSV must be a local file no larger than 128 MiB")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(set(header)) != len(header) or any(not name for name in header):
        raise WorkbenchError("INVALID_DATASET", "CSV header must contain unique, nonempty column names")
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise WorkbenchError("INVALID_DATASET", "CSV contains no data rows")
    numeric = frame.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise WorkbenchError("INVALID_DATASET", "Infinite numeric values must be cleaned explicitly")
    return frame


def inspect_csv(dataset_path: str, target: str | None = None) -> dict:
    path = Path(dataset_path).expanduser().resolve()
    frame = read_csv(path)
    if target is not None and target not in frame:
        raise WorkbenchError("INVALID_TARGET", "Target column does not exist")
    columns, warnings = [], []
    for name in frame:
        series = frame[name]
        unique = int(series.nunique(dropna=True))
        kind = "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"
        columns.append({"name": name, "kind": kind, "dtype": str(series.dtype),
                        "missing": int(series.isna().sum()), "unique": unique})
        if unique <= 1:
            warnings.append(f"{name}: constant or entirely missing")
        if name != target and unique == len(frame):
            warnings.append(f"{name}: all values unique; inspect whether this is an identifier")
    duplicate_rows = int(frame.duplicated().sum())
    if duplicate_rows:
        warnings.append("Duplicate rows exist; consider whether grouping is needed to avoid leakage")
    return {"sha256": sha256(path), "rows": len(frame), "columns": columns,
            "duplicate_rows": duplicate_rows, "target": target,
            "target_missing": int(frame[target].isna().sum()) if target else None,
            "warnings": warnings, "note": "Diagnostics do not automatically identify leakage or choose a task. No raw rows are returned."}


def make_split(frame: pd.DataFrame, spec: ExperimentSpec) -> tuple[dict, dict]:
    if spec.final_test_fraction:
        # Reserve the final holdout first. The second splitter only sees development rows.
        outer = spec.model_copy(update={"test_fraction": spec.final_test_fraction, "final_test_fraction": 0.0})
        first, details = make_split(frame, outer)
        development = frame.iloc[first["train"]].reset_index(drop=True)
        inner = spec.model_copy(update={"test_fraction": spec.test_fraction / (1 - spec.final_test_fraction),
                                       "final_test_fraction": 0.0})
        second, inner_details = make_split(development, inner)
        split = {key: [first["train"][i] for i in rows] for key, rows in second.items()}
        split["final_test"] = first["test"]
        return split, {"strategy": spec.split.strategy, "column": spec.split.column,
                       "train_rows": len(split["train"]), "test_rows": len(split["test"]),
                       "final_test_rows": len(split["final_test"]), "development": inner_details,
                       "final_boundary": details,
                       "note": "test denotes validation; final_test is evaluated only by finalize after selection"}
    labels = frame[spec.target]
    positions = np.arange(len(frame))
    strategy, column = spec.split.strategy, spec.split.column
    details = {"strategy": strategy, "column": column}
    if strategy != "random" and (column not in frame or frame[column].isna().any()):
        raise WorkbenchError("INVALID_SPLIT", "Split column must exist and contain no missing values")
    try:
        if strategy == "random":
            train, test = train_test_split(positions, test_size=spec.test_fraction, random_state=spec.seed,
                                          stratify=labels if spec.task == "classification" else None)
        elif strategy == "group":
            train, test = next(GroupShuffleSplit(n_splits=1, test_size=spec.test_fraction,
                              random_state=spec.seed).split(frame, groups=frame[column]))
            details.update(train_groups=int(frame[column].iloc[train].nunique()),
                           test_groups=int(frame[column].iloc[test].nunique()))
        else:
            if pd.api.types.is_numeric_dtype(frame[column]):
                raise ValueError("time column must contain ISO date/time strings, not ambiguous numeric timestamps")
            times = pd.to_datetime(frame[column], utc=True, errors="raise", format="mixed")
            if times.isna().any():
                raise ValueError("time column contains missing timestamps")
            order = np.argsort(times.to_numpy(), kind="stable")
            cut = len(frame) - math.ceil(len(frame) * spec.test_fraction)
            boundary = times.iloc[order[cut]]
            train = positions[(times < boundary).to_numpy()]
            test = positions[(times >= boundary).to_numpy()]
            details["validation_start"] = boundary.isoformat()
    except (ValueError, TypeError) as error:
        raise WorkbenchError("INVALID_SPLIT", str(error)) from error
    if len(train) < 2 or len(test) < 2:
        raise WorkbenchError("INVALID_SPLIT", "Training and validation sets must each contain at least two rows")
    if spec.task == "classification" and set(labels.iloc[train]) != set(labels):
        raise WorkbenchError("INVALID_SPLIT", "Training rows must contain every target class; adjust the split")
    details.update(train_rows=len(train), test_rows=len(test))
    return {"train": train.tolist(), "test": test.tolist()}, details
