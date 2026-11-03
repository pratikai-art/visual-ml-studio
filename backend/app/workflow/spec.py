"""Node specification, parameter validation and the node registry."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class NodeError(Exception):
    """An error with a beginner-friendly message and optional technical details."""

    def __init__(self, message: str, technical: str | None = None):
        super().__init__(message)
        self.message = message
        self.technical = technical


@dataclass
class Param:
    name: str
    label: str
    type: str  # int float bool select string text column columns
    default: Any = None
    description: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    options: Optional[list] = None
    nullable: bool = False
    advanced: bool = False
    map: Any = None  # dict or callable: UI value -> python value used by sklearn
    visible_if: Optional[dict] = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "name", "label", "type", "default", "description", "min", "max", "step",
            "options", "nullable", "advanced", "visible_if")}

    def py(self, v: Any) -> Any:
        if self.map is None:
            return v
        if callable(self.map):
            return self.map(v)
        return self.map.get(v, v)


@dataclass
class Port:
    name: str
    type: str  # dataset | split | trained | metrics | figure
    label: str = ""
    multi: bool = False
    required: bool = True

    def to_dict(self):
        return {"name": self.name, "type": self.type, "label": self.label or self.name,
                "multi": self.multi, "required": self.required}


@dataclass
class NodeResult:
    outputs: dict = field(default_factory=dict)
    preview: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


@dataclass
class NodeSpec:
    type: str
    name: str
    category: str
    description: str
    inputs: list
    outputs: list
    params: list
    run: Callable
    codegen: Callable
    subcategory: str = ""
    icon: str = "box"
    doc: dict = field(default_factory=dict)
    example: str = ""
    keywords: str = ""

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def to_dict(self) -> dict:
        return {
            "type": self.type, "name": self.name, "category": self.category,
            "subcategory": self.subcategory, "description": self.description,
            "icon": self.icon, "inputs": [p.to_dict() for p in self.inputs],
            "outputs": [p.to_dict() for p in self.outputs],
            "params": [p.to_dict() for p in self.params], "doc": self.doc,
            "example": self.example, "keywords": self.keywords,
            "compatibility": {
                "accepts": [p.type for p in self.inputs],
                "produces": [p.type for p in self.outputs],
            },
        }


REGISTRY: dict[str, NodeSpec] = {}

CATEGORY_ORDER = [
    "Data Input", "Data Exploration", "Data Cleaning", "Data Preprocessing",
    "Feature Selection", "Data Splitting", "Machine Learning", "Clustering",
    "Anomaly Detection",
    "Dimensionality Reduction", "Evaluation", "Visualization",
]


def register(spec: NodeSpec) -> NodeSpec:
    if spec.type in REGISTRY:
        raise ValueError(f"duplicate node type {spec.type}")
    REGISTRY[spec.type] = spec
    return spec


def validate_params(spec: NodeSpec, raw: dict | None) -> tuple[dict, list]:
    """Return (clean params with defaults filled in, list of error messages)."""
    raw = dict(raw or {})
    out: dict = {}
    errors: list = []
    for p in spec.params:
        v = raw.get(p.name, p.default)
        if p.type in ("int", "float"):
            if v is None or v == "":
                if p.nullable:
                    out[p.name] = None
                else:
                    errors.append(f"'{p.label}' needs a number.")
                continue
            try:
                if isinstance(v, bool):
                    raise ValueError
                if p.type == "int":
                    if float(v) != int(float(v)):
                        raise ValueError
                    v = int(float(v))
                else:
                    v = float(v)
            except (ValueError, TypeError):
                errors.append(f"'{p.label}' has an invalid value ({v!r}).")
                continue
            if p.min is not None and v < p.min:
                errors.append(f"'{p.label}' must be at least {p.min}.")
            if p.max is not None and v > p.max:
                errors.append(f"'{p.label}' must be at most {p.max}.")
        elif p.type == "bool":
            if v is None:
                v = bool(p.default)
            if isinstance(v, str):
                v = v.lower() in ("1", "true", "yes")
            v = bool(v)
        elif p.type == "select":
            if v is None:
                v = p.default
            if p.options and v not in p.options:
                errors.append(f"'{p.label}' must be one of: {', '.join(map(str, p.options))}.")
        elif p.type == "columns":
            if v is None:
                v = []
            if not isinstance(v, (list, tuple)):
                errors.append(f"'{p.label}' must be a list of column names.")
                continue
            v = [str(c) for c in v]
        else:  # string / text / column
            v = "" if v is None else str(v)
        out[p.name] = v
    return out, errors
