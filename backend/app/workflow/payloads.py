"""Runtime objects that flow along workflow edges."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd


@dataclass
class Dataset:
    df: pd.DataFrame
    target: Optional[str] = None
    task: Optional[str] = None  # classification | regression | clustering | None
    notes: list = field(default_factory=list)


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    task: str
    target: str
    seed: int
    time_ordered: bool = False
    steps: list = field(default_factory=list)  # preprocessing recipes, fitted ONLY inside pipelines
    fingerprint: str = ""

    def with_step(self, step: dict) -> "Split":
        return Split(self.X_train, self.X_test, self.y_train, self.y_test, self.task, self.target,
                     self.seed, self.time_ordered, self.steps + [step], self.fingerprint)


@dataclass
class Trained:
    pipeline: Any
    split: Split
    algo_key: str
    algo_name: str
    task: str
    kwargs: dict
    y_pred: Any
    y_proba: Any
    y_pred_train: Any
    y_proba_train: Any
    train_time: float
    infer_time: float
    cv: Optional[dict] = None
    search: Optional[dict] = None
    classes: Optional[list] = None
    feature_names: list = field(default_factory=list)
