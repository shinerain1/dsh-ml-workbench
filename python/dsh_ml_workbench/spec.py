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


class SplitSpec(StrictModel):
    strategy: Literal["random", "group", "time"] = "random"
    column: str | None = None

    @model_validator(mode="after")
    def validate_column(self):
        if (self.strategy == "random") != (self.column is None):
            raise ValueError("random split omits column; group/time split requires column")
        return self


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
    split: SplitSpec = Field(default_factory=SplitSpec)
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
        if self.split.column == self.target:
            raise ValueError("split.column cannot be the target")
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
    offset: int = Field(default=0, ge=0)
    state: Literal["created", "starting", "running", "cancelling", "succeeded", "failed", "interrupted", "cancelled"] | None = None
    model: str | None = None
    parent_run_id: str | None = None


class CompareRequest(StrictModel):
    run_ids: list[str] = Field(min_length=2, max_length=30)
    metric: str | None = None


class ExportRequest(RunRequest):
    include_data: bool = False


class InspectRequest(StrictModel):
    dataset_path: str
    target: str | None = None


class DiffRequest(StrictModel):
    left_run_id: str
    right_run_id: str


class PredictRequest(RunRequest):
    dataset_path: str


class SearchSpace(StrictModel):
    type: Literal["int", "float", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: list[str | int | float | bool | None] | None = None

    @model_validator(mode="after")
    def validate_distribution(self):
        if self.type == "categorical":
            if not self.choices or len(self.choices) > 100 or self.low is not None or self.high is not None or self.step is not None or self.log:
                raise ValueError("categorical requires 1-100 choices and no numeric distribution fields")
        else:
            if self.low is None or self.high is None or self.low > self.high or self.choices is not None:
                raise ValueError("numeric search requires low <= high and no choices")
            if self.log and (self.low <= 0 or self.step is not None):
                raise ValueError("log distribution requires low > 0 and no step")
            if self.step is not None and self.step <= 0:
                raise ValueError("step must be positive")
            if self.type == "int" and any(value is not None and type(value) is not int for value in (self.low, self.high, self.step)):
                raise ValueError("int bounds and step must be integers")
        return self


class SearchRequest(StrictModel):
    study_name: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")


class SearchCreateRequest(SearchRequest):
    run_id: str
    search_space: dict[str, SearchSpace] = Field(min_length=1, max_length=20)
    n_trials: int = Field(default=10, ge=1, le=100)
    metric: str | None = None
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    reuse_preprocessing: bool = False
    trial_timeout_seconds: int = Field(default=600, ge=5, le=86400)

    @model_validator(mode="after")
    def validate_parameter_names(self):
        if {"random_state", "n_jobs"} & self.search_space.keys():
            raise ValueError("search_space cannot override random_state or n_jobs")
        return self


class RpcRequest(StrictModel):
    protocol_version: Literal[1] = 1
    operation: Literal["health", "create", "start", "get", "list", "cancel", "branch", "compare", "export",
                       "inspect", "diff", "predict", "search_create", "search_step", "search_get", "search_cancel"]
    args: dict[str, Any] = Field(default_factory=dict)


class WorkbenchError(Exception):
    """Stable error identity for model-facing tools and command-line clients."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
