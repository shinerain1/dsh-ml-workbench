"""Versioned inputs for the local workflow; MLflow owns tracking records."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Preprocessing(StrictModel):
    numeric_imputer: Literal["median", "mean", "most_frequent"] = "median"
    scale_numeric: bool = True
    max_categories: int = Field(default=64, ge=2, le=4096)
    drop_columns: list[str] = Field(default_factory=list)


class ExperimentSpec(StrictModel):
    dataset_path: str = Field(min_length=1)
    target: str = Field(min_length=1)
    task: Literal["classification", "regression"]
    name: str = Field(default="experiment", min_length=1, max_length=160)
    experiment_name: str = Field(default="DSH ML workbench", min_length=1, max_length=160)
    model: str = "random_forest"
    params: dict[str, Any] = Field(default_factory=dict)
    preprocessing: Preprocessing = Field(default_factory=Preprocessing)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    test_fraction: float = Field(default=0.2, gt=0.0, lt=0.5)
    preprocessor_path: str | None = None
    estimator_path: str | None = None

    @model_validator(mode="after")
    def validate_model(self):
        names = {"random_forest", "extra_trees", "hist_gradient_boosting", "custom"}
        names.add("logistic_regression" if self.task == "classification" else "ridge")
        if self.model not in names:
            raise ValueError(f"model must be one of {sorted(names)} for {self.task}")
        if (self.model == "custom") != (self.estimator_path is not None):
            raise ValueError("custom model requires estimator_path; other models must omit it")
        if {"random_state", "n_jobs"} & self.params.keys():
            raise ValueError("use seed instead of random_state; n_jobs is fixed at 1")
        if self.target in self.preprocessing.drop_columns:
            raise ValueError("drop_columns must not contain the target")
        if len(set(self.preprocessing.drop_columns)) != len(self.preprocessing.drop_columns):
            raise ValueError("drop_columns must not contain duplicates")
        return self


class CreateRequest(ExperimentSpec):
    request_key: str | None = Field(default=None, min_length=1, max_length=256)


class BranchRequest(StrictModel):
    run_id: str
    changes: dict[str, Any] = Field(default_factory=dict)
    reuse_preprocessing: bool = True
    request_key: str | None = Field(default=None, min_length=1, max_length=256)


class RunRequest(StrictModel):
    run_id: str


class ListRequest(StrictModel):
    limit: int = Field(default=20, ge=1, le=100)


class CompareRequest(StrictModel):
    run_ids: list[str] = Field(min_length=2, max_length=30)
    metric: str | None = None


class ExportRequest(RunRequest):
    include_data: bool = False


class RpcRequest(StrictModel):
    protocol_version: Literal[1] = 1
    operation: Literal["health", "create", "start", "get", "list", "cancel", "branch", "compare", "export"]
    args: dict[str, Any] = Field(default_factory=dict)


class WorkbenchError(Exception):
    """Stable error identity for model-facing tools and command-line clients."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
