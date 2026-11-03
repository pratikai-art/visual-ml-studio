import ast, sys, nbformat, pytest
from app.codegen.generator import to_notebook, to_script, validate_notebook
from app.workflow.graph import Workflow
from tests.helpers import N, E, run


def student_wf():
    return {"name": "Student Score Regression", "journal": {"hypothesis": "More study hours raise exam scores."},
            "nodes": [N("d", "data_input", builtin="student_scores"), N("x", "data_explorer"), N("t", "select_target", target="Exam_Score", task="regression"),
                      N("s", "train_test_split"), N("im", "pp_imputer"), N("sc", "pp_scaler"), N("m", "reg_random_forest", n_estimators=40, cv_folds=3),
                      N("ev", "evaluate"), N("c", "chart_model", chart="actual_vs_predicted"), N("fi", "chart_model", chart="feature_importance"),
                      N("m2", "reg_ridge", search="grid", search_space='{"alpha": [0.1, 1.0, 10.0]}'), N("ev2", "evaluate"), N("cmp", "compare_models"), N("hist", "chart_dataset", chart="correlation_heatmap")],
            "edges": [E("d", "dataset", "x", "dataset"), E("x", "dataset", "t", "dataset"), E("t", "dataset", "s", "dataset"), E("s", "split", "im", "split"), E("im", "split", "sc", "split"),
                      E("sc", "split", "m", "split"), E("m", "trained", "ev", "trained"), E("m", "trained", "c", "trained"), E("m", "trained", "fi", "trained"),
                      E("sc", "split", "m2", "split"), E("m2", "trained", "ev2", "trained"), E("ev", "metrics", "cmp", "metrics"), E("ev2", "metrics", "cmp", "metrics"), E("x", "dataset", "hist", "dataset")]}


def test_notebook_valid_structure_and_order():
    nb = to_notebook(Workflow(**student_wf()))
    rep = validate_notebook(nb)
    assert rep["valid"], rep
    sources = [c.source for c in nb.cells]
    pos = {k: next(i for i, s in enumerate(sources) if k in s) for k in ("Required libraries", "Dataset loading", "Train/test split", "Train: Random Forest Regressor", "Evaluation:", "Conclusions", "Reproducibility")}
    assert list(pos.values()) == sorted(pos.values())
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "n_estimators=40" in text and "RandomForestRegressor" in text and "GridSearchCV" in text and "More study hours" in "\n".join(sources)
    assert "/tmp" not in text and "uploads" not in text  # no app-local paths


def test_generated_script_executes_and_matches_app_metrics(monkeypatch):
    import plotly.basedatatypes as b
    monkeypatch.setattr(b.BaseFigure, "show", lambda *a, **k: None)
    wf = student_wf()
    job = run(wf)
    app_metrics = job["nodes"]["ev"]["preview"]["metrics"]
    ns = {}
    exec(compile(to_script(Workflow(**wf)), "<generated>", "exec"), ns)   # independent of app state
    gen = ns["metrics_n8"]
    for k in ("mae", "rmse", "r2"):
        assert gen[k] == pytest.approx(app_metrics[k], rel=1e-9)


def test_classification_and_clustering_scripts_execute(monkeypatch):
    import plotly.basedatatypes as b
    monkeypatch.setattr(b.BaseFigure, "show", lambda *a, **k: None)
    wf = {"nodes": [N("d", "data_input", builtin="breast_cancer"), N("t", "select_target", target="target"), N("s", "train_test_split"), N("sc", "pp_scaler", method="robust"),
                    N("fs", "pp_feature_select", k=8), N("m", "clf_svc", C=2.0), N("ev", "evaluate", average="weighted"), N("cm", "chart_model", chart="confusion_matrix", normalize=True),
                    N("roc", "chart_model", chart="roc_curve"), N("k", "clustering", algorithm="dbscan", eps=1.2, label_column="target"), N("pc", "pca"), N("sc2", "chart_dataset", chart="scatter", x="PC1", y="PC2")],
          "edges": [E("d", "dataset", "t", "dataset"), E("t", "dataset", "s", "dataset"), E("s", "split", "sc", "split"), E("sc", "split", "fs", "split"), E("fs", "split", "m", "split"),
                    E("m", "trained", "ev", "trained"), E("m", "trained", "cm", "trained"), E("m", "trained", "roc", "trained"), E("d", "dataset", "k", "dataset"), E("k", "dataset", "pc", "dataset"), E("pc", "dataset", "sc2", "dataset")]}
    job = run(wf)
    assert job["status"] == "Completed", {k: v["error"] for k, v in job["nodes"].items() if v["error"]}
    ns = {}
    exec(compile(to_script(Workflow(**wf)), "<generated>", "exec"), ns)
    assert ns["metrics_n7"]["accuracy"] == pytest.approx(job["nodes"]["ev"]["preview"]["metrics"]["accuracy"])


def test_anomaly_detection_script_executes():
    wf = {"nodes": [N("d", "data_input", builtin="synthetic_clusters", rows=120),
                    N("a", "anomaly_detection", algorithm="isolation_forest", n_estimators=30)],
          "edges": [E("d", "dataset", "a", "dataset")]}
    code = to_script(Workflow(**wf))
    ns = {}
    exec(compile(code, "<generated>", "exec"), ns)
    assert "df_n2" in ns and "anomaly" in ns["df_n2"].columns


@pytest.mark.parametrize("node_type", ["tsne", "truncated_svd"])
def test_dimensionality_reduction_scripts_execute(node_type):
    wf = {"nodes": [N("d", "data_input", builtin="wine"),
                    N("r", node_type, n_components=2, perplexity=10)],
          "edges": [E("d", "dataset", "r", "dataset")]}
    ns = {}
    exec(compile(to_script(Workflow(**wf)), "<generated>", "exec"), ns)
    assert "df_n2" in ns


def test_pytorch_script_executes():
    wf = {"nodes": [
        N("d", "data_input", builtin="iris"),
        N("t", "select_target", target="target", task="classification"),
        N("s", "train_test_split", test_size=0.25),
        N("m", "pytorch_neural_network", architecture="gru", epochs=2, hidden=16),
    ], "edges": [
        E("d", "dataset", "t", "dataset"), E("t", "dataset", "s", "dataset"),
        E("s", "split", "m", "split"),
    ]}
    ns = {}
    exec(compile(to_script(Workflow(**wf)), "<generated>", "exec"), ns)
    assert "y_pred_n4" in ns and len(ns["y_pred_n4"]) > 0


def test_notebook_executes_in_fresh_kernel(tmp_path):
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb = to_notebook(Workflow(**student_wf()))
    try:
        nbclient.NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}).execute()
    except Exception as e:  # kernel spec missing in CI
        if "No such kernel" in str(e):
            pytest.skip("python3 kernel not installed")
        raise
    assert not [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]


def test_upload_notebook_has_dataset_placeholder_not_local_path():
    import numpy as np, pandas as pd
    from fastapi.testclient import TestClient
    from app.main import app
    df = pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0) ** 2}); df["y"] = df.a * 3
    r = TestClient(app).post("/api/datasets", files={"file": ("my data.csv", df.to_csv(index=False).encode(), "text/csv")}).json()
    wf = {"nodes": [N("d", "data_input", source="upload", dataset_id=r["meta"]["id"]), N("t", "select_target", target="y", task="regression")], "edges": [E("d", "dataset", "t", "dataset")]}
    code = "\n".join(c.source for c in to_notebook(Workflow(**wf)).cells if c.cell_type == "code")
    assert "DATASET_PATH = Path('my_data.csv')" in code and "uploads" not in code and "files.upload()" in code


def test_invalid_workflow_refuses_codegen():
    from app.workflow.spec import NodeError
    with pytest.raises(NodeError):
        to_notebook(Workflow(nodes=[N("s", "train_test_split")]))
