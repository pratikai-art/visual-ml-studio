"""Preprocessing 'steps': one declarative plan -> (a) sklearn object, (b) Python source.
Steps are fitted only inside sklearn Pipelines on training data (no leakage)."""
from __future__ import annotations
import importlib
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline

from ..workflow.spec import NodeError

NAMES = {  # names usable inside Raw() expressions -> import line
    "np": "import numpy as np",
    "f_regression": "from sklearn.feature_selection import f_regression",
    "f_classif": "from sklearn.feature_selection import f_classif",
    "mutual_info_regression": "from sklearn.feature_selection import mutual_info_regression",
    "mutual_info_classif": "from sklearn.feature_selection import mutual_info_classif",
    "chi2": "from sklearn.feature_selection import chi2",
    "RandomForestRegressor": "from sklearn.ensemble import RandomForestRegressor",
    "RandomForestClassifier": "from sklearn.ensemble import RandomForestClassifier",
    "LinearRegression": "from sklearn.linear_model import LinearRegression",
    "LogisticRegression": "from sklearn.linear_model import LogisticRegression",
}


class Raw:
    """A python expression built only from internal constants (never user text)."""
    def __init__(self, expr: str, names=()):
        self.expr, self.names = expr, tuple(names)

    def __repr__(self):
        return self.expr

    def value(self):
        import sklearn.ensemble, sklearn.feature_selection, sklearn.linear_model  # noqa: F401
        ns = {"np": np}
        for n, imp in NAMES.items():
            if n != "np":
                mod, nm = imp.split(" import ")[0].replace("from ", ""), n
                ns[nm] = getattr(importlib.import_module(mod), nm)
        return eval(self.expr, {"__builtins__": {}}, ns)  # noqa: S307 - internal expr only


def _imp(path: str):
    mod, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def step_plan(step: dict, task: str) -> dict:
    kind, p = step["kind"], step["params"]
    cols = p.get("columns") or []
    if kind == "imputer":
        s = p["strategy"]
        which = p["apply_to"]
        if s in ("mean", "median", "knn") and which != "numeric":
            raise NodeError(f"'{s}' imputation only works on numerical columns. Set 'Apply to' = numeric, or use most_frequent/constant.")
        if s == "knn":
            return dict(cls="sklearn.impute.KNNImputer", kw={"n_neighbors": p["n_neighbors"]}, apply=which, columns=cols)
        kw = {"strategy": s}
        if s == "constant":
            fv = p["fill_value"]
            try:
                kw["fill_value"] = float(fv) if which == "numeric" else str(fv)
            except ValueError:
                raise NodeError(f"Fill value '{fv}' is not a number but numerical columns were selected.")
        return dict(cls="sklearn.impute.SimpleImputer", kw=kw, apply=which, columns=cols)
    if kind == "scaler":
        m = p["method"]
        table = {
            "standard": ("sklearn.preprocessing.StandardScaler", {}),
            "minmax": ("sklearn.preprocessing.MinMaxScaler", {}),
            "robust": ("sklearn.preprocessing.RobustScaler", {}),
            "maxabs": ("sklearn.preprocessing.MaxAbsScaler", {}),
            "power": ("sklearn.preprocessing.PowerTransformer", {}),
            "quantile": ("sklearn.preprocessing.QuantileTransformer", {"n_quantiles": 100, "random_state": step.get("seed", 42)}),
            "normalizer": ("sklearn.preprocessing.Normalizer", {}),
            "polynomial": ("sklearn.preprocessing.PolynomialFeatures", {"degree": p["degree"], "include_bias": False}),
            "log1p": ("sklearn.preprocessing.FunctionTransformer", {"func": Raw("np.log1p", ["np"]), "feature_names_out": "one-to-one"}),
        }
        c, kw = table[m]
        return dict(cls=c, kw=kw, apply="numeric", columns=cols)
    if kind == "encoder":
        if p["method"] == "onehot":
            kw = {"handle_unknown": "ignore", "sparse_output": False}
            if p["drop_first"]:
                kw["drop"] = "first"
            return dict(cls="sklearn.preprocessing.OneHotEncoder", kw=kw, apply="categorical", columns=cols)
        return dict(cls="sklearn.preprocessing.OrdinalEncoder",
                    kw={"handle_unknown": "use_encoded_value", "unknown_value": -1}, apply="categorical", columns=cols)
    if kind == "feature_select":
        m = p["method"]
        reg = task == "regression"
        if m == "select_k_best":
            sc = p["score"]
            if sc == "chi2":
                if reg:
                    raise NodeError("Chi-square only works for classification targets and non-negative features.")
                fn = Raw("chi2", ["chi2"])
            elif sc == "mutual_info":
                n = "mutual_info_regression" if reg else "mutual_info_classif"
                fn = Raw(n, [n])
            else:
                n = "f_regression" if reg else "f_classif"
                fn = Raw(n, [n])
            return dict(cls="sklearn.feature_selection.SelectKBest", kw={"score_func": fn, "k": p["k"]}, apply=None, columns=[])
        if m == "variance_threshold":
            return dict(cls="sklearn.feature_selection.VarianceThreshold", kw={"threshold": p["threshold"]}, apply=None, columns=[])
        est = "RandomForestRegressor" if reg else "RandomForestClassifier"
        e = Raw(f"{est}(n_estimators=100, random_state={step.get('seed', 42)})", [est])
        if m == "select_from_model":
            return dict(cls="sklearn.feature_selection.SelectFromModel", kw={"estimator": e, "threshold": "median"}, apply=None, columns=[])
        lin = Raw("LinearRegression()", ["LinearRegression"]) if reg else Raw("LogisticRegression(max_iter=1000)", ["LogisticRegression"])
        return dict(cls="sklearn.feature_selection.RFE", kw={"estimator": lin, "n_features_to_select": p["k"]}, apply=None, columns=[])
    raise NodeError(f"Unknown preprocessing step '{kind}'.")


def _selector_arg(plan):
    if plan["columns"]:
        return list(plan["columns"])
    return ("number", None) if plan["apply"] == "numeric" else (None, "number") if plan["apply"] == "categorical" else (None, None)


def build_step(step: dict, task: str):
    plan = step_plan(step, task)
    kw = {k: (v.value() if isinstance(v, Raw) else v) for k, v in plan["kw"].items()}
    obj = _imp(plan["cls"])(**kw)
    if plan["apply"] is None:
        return obj.set_output(transform="pandas")
    sel = _selector_arg(plan)
    if isinstance(sel, tuple):
        sel = make_column_selector(dtype_include=sel[0], dtype_exclude=sel[1])
    ct = ColumnTransformer([(step["kind"], obj, sel)], remainder="passthrough", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def build_pipeline(steps: list, task: str, final=None) -> Pipeline:
    items = [(f"s{i}_{s['kind']}", build_step(s, task)) for i, s in enumerate(steps)]
    if final is not None:
        items.append(("model", final))
    if not items:
        items = [("passthrough", __import__("sklearn.preprocessing", fromlist=["FunctionTransformer"]).FunctionTransformer(
            feature_names_out="one-to-one").set_output(transform="pandas"))]
    return Pipeline(items)


def step_code(step: dict, task: str) -> tuple[str, set]:
    """Return (python expression for the step object, imports)."""
    plan = step_plan(step, task)
    imports, cls = set(), plan["cls"]
    mod, name = cls.rsplit(".", 1)
    imports.add(f"from {mod} import {name}")
    for v in plan["kw"].values():
        if isinstance(v, Raw):
            for n in v.names:
                imports.add(NAMES[n])
    kws = ", ".join(f"{k}={v!r}" for k, v in plan["kw"].items())
    obj = f"{name}({kws})"
    if plan["apply"] is None:
        return f'{obj}.set_output(transform="pandas")', imports
    imports.add("from sklearn.compose import ColumnTransformer, make_column_selector")
    sel = _selector_arg(plan)
    if isinstance(sel, list):
        sel_code = repr(sel)
    else:
        sel_code = f"make_column_selector(dtype_include={sel[0]!r}, dtype_exclude={sel[1]!r})"
    return (f'ColumnTransformer([({step["kind"]!r}, {obj}, {sel_code})], remainder="passthrough", '
            f'verbose_feature_names_out=False).set_output(transform="pandas")'), imports
