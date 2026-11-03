import numpy as np, pandas as pd, pytest
from app import nodes  # noqa
from app.workflow.spec import REGISTRY, validate_params
from app.workflow.validator import check_connection, validate
from app.workflow.graph import Workflow, topo_order, CycleError
from app.workflow.cache import ExecutionCache
from app.nodes.steps import build_pipeline, step_code
from app.nodes.evaluation import cls_metrics, reg_metrics
from app.workflow.executor import friendly
from tests.helpers import N, E, run


def test_registry_has_real_algorithms():
    types = set(REGISTRY)
    for t in ("clf_random_forest", "clf_logistic_regression", "reg_ridge", "reg_random_forest", "clustering", "pca", "evaluate", "train_test_split"):
        assert t in types
    assert sum(t.startswith(("clf_", "reg_")) for t in types) >= 28


def test_param_validation_ranges_and_options():
    spec = REGISTRY["reg_random_forest"]
    p, errs = validate_params(spec, {"n_estimators": 0})
    assert any("at least" in e for e in errs)
    _, errs = validate_params(spec, {"n_estimators": "abc"})
    assert errs
    p, errs = validate_params(REGISTRY["clf_logistic_regression"], {"solver": "nope"})
    assert any("must be one of" in e for e in errs)
    p, errs = validate_params(spec, {})
    assert not errs and p["max_depth"] is None  # nullable default


def test_connection_rules():
    assert check_connection("data_input", "dataset", "train_test_split", "dataset") is None
    why = check_connection("data_input", "dataset", "reg_ridge", "split")
    assert why and "Train/Test Split" in why
    assert check_connection("reg_ridge", "trained", "evaluate", "trained") is None
    assert check_connection("evaluate", "metrics", "evaluate", "trained")


def test_cycle_detection_and_missing_input():
    wf = Workflow(nodes=[N("a", "drop_duplicates"), N("b", "drop_duplicates")],
                  edges=[E("a", "dataset", "b", "dataset"), E("b", "dataset", "a", "dataset")])
    with pytest.raises(CycleError):
        topo_order(wf)
    assert any("cycle" in i["message"].lower() for i in validate(wf))
    wf2 = Workflow(nodes=[N("s", "train_test_split")])
    assert any("Missing required input" in i["message"] for i in validate(wf2))


def test_single_connection_port():
    wf = Workflow(nodes=[N("a", "data_input"), N("b", "data_input"), N("s", "train_test_split")],
                  edges=[E("a", "dataset", "s", "dataset"), E("b", "dataset", "s", "dataset")])
    assert any("only one connection" in i["message"] for i in validate(wf))


def test_cache_key_changes_with_params_and_upstream():
    k1 = ExecutionCache.key("t", {"a": 1}, ["x"])
    assert k1 == ExecutionCache.key("t", {"a": 1}, ["x"])
    assert k1 != ExecutionCache.key("t", {"a": 2}, ["x"])
    assert k1 != ExecutionCache.key("t", {"a": 1}, ["y"])


def test_regression_metrics_exact():
    y = np.array([3.0, 5.0, 7.0, 9.0]); p = np.array([2.0, 5.0, 8.0, 9.0])
    m, _ = reg_metrics(y, p, 1)
    assert m["mae"] == pytest.approx(0.5) and m["mse"] == pytest.approx(0.5) and m["rmse"] == pytest.approx(0.5 ** 0.5)
    assert m["r2"] == pytest.approx(1 - 2 / 20)


def test_mape_zero_warning():
    m, w = reg_metrics(np.array([0.0, 1.0, 2.0]), np.array([0.1, 1.0, 2.0]), 1)
    assert "mape" not in m and any("MAPE" in x for x in w)


def test_classification_metrics_exact():
    y = np.array([0, 0, 1, 1]); p = np.array([0, 1, 1, 1])
    m, _ = cls_metrics(y, p, None, [0, 1], "binary", "1")
    assert m["accuracy"] == 0.75 and m["precision"] == pytest.approx(2 / 3) and m["recall"] == 1.0


def test_scaler_fit_on_train_only_no_leakage():
    rng = np.random.default_rng(0)
    Xtr = pd.DataFrame({"a": rng.normal(0, 1, 50)}); Xte = pd.DataFrame({"a": rng.normal(100, 1, 20)})
    pipe = build_pipeline([{"kind": "scaler", "params": {"method": "standard", "degree": 2, "columns": []}, "seed": 1}], "regression")
    pipe.fit(Xtr)
    out = pipe.transform(Xte)
    assert out["a"].mean() > 50  # would be ~0 if the scaler had seen the test data
    scaler = pipe.steps[0][1].named_transformers_["scaler"]
    assert scaler.mean_[0] == pytest.approx(Xtr["a"].mean())


def test_step_codegen_matches_plan():
    code, imps = step_code({"kind": "imputer", "params": {"strategy": "median", "apply_to": "numeric", "columns": [], "fill_value": "0", "n_neighbors": 5}}, "regression")
    assert "SimpleImputer(strategy='median')" in code and "from sklearn.impute import SimpleImputer" in imps


def test_friendly_errors():
    assert "not been encoded" in friendly(ValueError("could not convert string to float: 'a'")).message
    assert "missing values" in friendly(ValueError("Input X contains NaN.")).message
