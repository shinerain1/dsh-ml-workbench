"""Portable scikit-learn training recipe, copied unchanged into every experiment.

Factories customize preprocessing and estimation; the fixed evaluator always uses
the stored validation rows. This module also runs standalone inside an export.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

# Avoid CPU oversubscription and preserve a documented execution environment.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["MLFLOW_ENABLE_TELEMETRY"] = "false"

import cloudpickle
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EVALUATOR_VERSION = "holdout-v1"


def load_factory(path: Path, function: str):
    name = "mlwb_custom_" + hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load custom factory {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    cloudpickle.register_pickle_by_value(module)
    factory = getattr(module, function, None)
    if not callable(factory):
        raise ValueError(f"{path.name} must define {function}")
    return factory


def build_preprocessor(frame: pd.DataFrame, config: dict):
    numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [name for name in frame.columns if name not in numeric]
    transformers = []
    if numeric:
        steps = [("impute", SimpleImputer(strategy=config["numeric_imputer"], keep_empty_features=True))]
        if config["scale_numeric"]:
            steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                      max_categories=config["max_categories"])),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop")


def build_estimator(spec: dict, inputs: Path):
    if spec["model"] == "custom":
        return load_factory(inputs / "estimator.py", "make_estimator")(spec["task"], spec["params"], spec["seed"])
    classification = spec["task"] == "classification"
    factories = {
        "random_forest": RandomForestClassifier if classification else RandomForestRegressor,
        "extra_trees": ExtraTreesClassifier if classification else ExtraTreesRegressor,
        "hist_gradient_boosting": HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor,
        "logistic_regression": LogisticRegression,
        "ridge": Ridge,
    }
    params = dict(spec["params"])
    params["random_state"] = spec["seed"]
    if spec["model"] in {"random_forest", "extra_trees"}:
        params.setdefault("n_estimators", 100)
        params["n_jobs"] = 1
    if spec["model"] == "logistic_regression":
        params.setdefault("max_iter", 1000)
    return factories[spec["model"]](**params)


def evaluate(task: str, truth, predicted) -> dict[str, float]:
    if task == "classification":
        result = {"accuracy": accuracy_score(truth, predicted),
                  "balanced_accuracy": balanced_accuracy_score(truth, predicted),
                  "f1_macro": f1_score(truth, predicted, average="macro", zero_division=0)}
    else:
        result = {"rmse": np.sqrt(np.mean((np.asarray(truth) - np.asarray(predicted)) ** 2)),
                  "mae": mean_absolute_error(truth, predicted), "r2": r2_score(truth, predicted)}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Evaluation produced non-finite metrics")
    return {key: float(value) for key, value in result.items()}


def train(inputs: Path, output: Path, data: Path | None = None) -> dict:
    config = json.loads((inputs / "config.json").read_text(encoding="utf-8"))
    split = json.loads((inputs / "split.json").read_text(encoding="utf-8"))
    dataset = data if data is not None else inputs / "dataset.csv"
    expected_hash = json.loads((inputs / "dataset.json").read_text(encoding="utf-8"))["sha256"]
    with dataset.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
            raise ValueError("Dataset hash differs from the recorded experiment")
    frame = pd.read_csv(dataset, low_memory=False)
    features = frame.drop(columns=[config["target"], *config["preprocessing"]["drop_columns"]])
    target = frame[config["target"]]
    x_train, x_test = features.iloc[split["train"]], features.iloc[split["test"]]
    y_train, y_test = target.iloc[split["train"]], target.iloc[split["test"]]
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if (inputs / "reused_preprocessor.pkl").exists():
        with (inputs / "reused_preprocessor.pkl").open("rb") as stream:
            preprocessor = cloudpickle.load(stream)
        transformed_train = preprocessor.transform(x_train)
    else:
        preprocessor = (load_factory(inputs / "preprocessor.py", "make_preprocessor")
                        (x_train.copy(), config["preprocessing"]) if (inputs / "preprocessor.py").exists()
                        else build_preprocessor(x_train, config["preprocessing"]))
        transformed_train = preprocessor.fit_transform(x_train, y_train)
    estimator = build_estimator(config, inputs)
    estimator.fit(transformed_train, y_train)
    pipeline = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
    train_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    predicted = pipeline.predict(x_test)
    predict_seconds = time.perf_counter() - prediction_started
    metrics = evaluate(config["task"], y_test, predicted)
    pd.DataFrame({"row_index": split["test"], "actual": y_test.to_numpy(), "prediction": predicted}).to_csv(
        output / "predictions.csv", index=False)
    with (output / "preprocessor.pkl").open("wb") as stream:
        cloudpickle.dump(preprocessor, stream)
    mlflow.sklearn.save_model(
        pipeline, str(output / "model"), signature=infer_signature(x_train, pipeline.predict(x_train)),
        pip_requirements=str(inputs / "requirements.txt"), serialization_format="cloudpickle",
    )
    result = {"metrics": metrics, "train_seconds": train_seconds, "predict_seconds": predict_seconds,
              "train_rows": len(x_train), "test_rows": len(x_test), "features": features.columns.tolist(),
              "reused_preprocessing": (inputs / "reused_preprocessor.pkl").exists(),
              "evaluator_version": EVALUATOR_VERSION}
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data", type=Path)
    options = parser.parse_args()
    train(options.inputs, options.output, options.data)
