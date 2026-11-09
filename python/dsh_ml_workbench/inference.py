"""Schema-checked local batch inference from a verified MLflow pipeline."""

import shutil
import uuid
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

from .datasets import read_csv
from .files import atomic_json, read_json, sha256
from .spec import WorkbenchError


def predict_csv(workbench, record: dict, dataset_path: str) -> dict:
    source = Path(dataset_path).expanduser().resolve()
    # Validate before creating the inference directory, then read its fixed copy.
    read_csv(source)
    directory = workbench.root / "predictions" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    try:
        shutil.copyfile(source, directory / "input.csv")
        frame = read_csv(directory / "input.csv")
        run_dir = workbench.run_dir(record["run_id"])
        result = read_json(run_dir / "output" / "result.json")
        features = result["features"]
        missing = set(features) - set(frame)
        if missing:
            raise WorkbenchError("FEATURE_SCHEMA_MISMATCH", "Missing features: " + ", ".join(sorted(missing)))
        original = read_csv(run_dir / "inputs" / "dataset.csv")
        numeric = set(original[features].select_dtypes(include=[np.number]).columns)
        for name in numeric:
            try:
                frame[name] = pd.to_numeric(frame[name], errors="raise")
            except (ValueError, TypeError) as error:
                raise WorkbenchError("FEATURE_SCHEMA_MISMATCH", f"{name} requires numeric values") from error
        if record["custom_code"] and not workbench.allow_custom_code:
            raise WorkbenchError("CUSTOM_CODE_DISABLED", "Enable allowCustomCode for this model")
        model = mlflow.sklearn.load_model(str(run_dir / "output" / "model"))
        selected = frame[features]
        predicted = model.predict(selected)
        output = pd.DataFrame({"row_index": range(len(frame)), "prediction": predicted})
        classes = None
        if record["task"] == "classification" and hasattr(model, "predict_proba"):
            classes = np.asarray(model.classes_).tolist()
            probabilities = model.predict_proba(selected)
            for index in range(len(classes)):
                output[f"probability_{index}"] = probabilities[:, index]
        output.to_csv(directory / "predictions.csv", index=False)
        manifest = {"run_id": record["run_id"], "rows": len(frame),
                    "input_sha256": sha256(directory / "input.csv"), "model_output_hash": record["output_hash"],
                    "prediction_sha256": sha256(directory / "predictions.csv"),
                    "classes": classes, "ignored_columns": sorted(set(frame) - set(features)),
                    "path": str(directory / "predictions.csv")}
        atomic_json(directory / "manifest.json", manifest)
        return manifest
    except BaseException:
        shutil.rmtree(directory)
        raise
