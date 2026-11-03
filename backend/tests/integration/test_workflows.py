import io, json, threading, time
import numpy as np, pandas as pd, pytest
from fastapi.testclient import TestClient
from app.main import app
from tests.helpers import N, E, run

client = TestClient(app)


def csv_bytes(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "city": rng.choice(["a", "b", "c"], n)})
    df.loc[rng.random(n) < 0.1, "x1"] = np.nan
    df["label"] = ((df["x2"] + (df["city"] == "a") + rng.normal(0, .3, n)) > 0.5).astype(int)
    return df.to_csv(index=False).encode()


def test_quoted_semicolon_csv_exposes_all_columns_and_target():
    content = (
        b'"fixed acidity";"volatile acidity";"alcohol";"quality"\n'
        b'"7.4";"0.70";"9.4";"5"\n'
        b'"7.8";"0.88";"9.8";"6"\n'
    )
    response = client.post("/api/datasets", files={"file": ("wine.csv", content, "text/csv")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["meta"]["delimiter"] == ";"
    assert body["profile"]["cols"] == 4
    assert [column["name"] for column in body["profile"]["columns"]] == [
        "fixed acidity", "volatile acidity", "alcohol", "quality"
    ]
    assert "quality" in body["profile"]["possible_targets"]


def test_csv_imputation_scaling_split_logreg_eval():
    r = client.post("/api/datasets", files={"file": ("d.csv", csv_bytes(), "text/csv")})
    assert r.status_code == 200, r.text
    ds = r.json()
    assert ds["profile"]["rows"] == 200 and ds["profile"]["columns"][0]["missing"] > 0
    wf = {"nodes": [N("d", "data_input", source="upload", dataset_id=ds["meta"]["id"]), N("t", "select_target", target="label", task="classification"),
                    N("s", "train_test_split", test_size=0.25), N("i1", "pp_imputer", strategy="median"), N("i2", "pp_imputer", strategy="most_frequent", apply_to="categorical"),
                    N("e", "pp_encoder"), N("sc", "pp_scaler"), N("m", "clf_logistic_regression"), N("ev", "evaluate"), N("roc", "chart_model", chart="roc_curve"),
                    N("cm", "chart_model", chart="confusion_matrix")],
          "edges": [E("d", "dataset", "t", "dataset"), E("t", "dataset", "s", "dataset"), E("s", "split", "i1", "split"), E("i1", "split", "i2", "split"), E("i2", "split", "e", "split"),
                    E("e", "split", "sc", "split"), E("sc", "split", "m", "split"), E("m", "trained", "ev", "trained"), E("m", "trained", "roc", "trained"), E("m", "trained", "cm", "trained")]}
    job = run(wf)
    assert job["status"] == "Completed", {k: v["error"] for k, v in job["nodes"].items() if v["error"]}
    met = job["nodes"]["ev"]["preview"]["metrics"]
    assert met["accuracy"] > 0.7 and 0 <= met["roc_auc"] <= 1
    fig = job["nodes"]["roc"]["preview"]["figure"]
    assert fig["data"] and len(fig["data"][0]["x"]) > 3  # real ROC points
    assert job["nodes"]["sc"]["preview"]["missing_after"] == 0


def test_categorical_not_encoded_friendly_error():
    wf = {"nodes": [N("d", "data_input", source="builtin", builtin="student_scores"), N("t", "select_target", target="Exam_Score", task="regression"),
                    N("s", "train_test_split"), N("m", "reg_ridge"), N("ev", "evaluate")],
          "edges": [E("d", "dataset", "t", "dataset"), E("t", "dataset", "s", "dataset"), E("s", "split", "m", "split"), E("m", "trained", "ev", "trained")]}
    job = run(wf)  # missing values, no imputer
    assert job["nodes"]["m"]["status"] == "failed"
    assert "missing values" in job["nodes"]["m"]["error"]["message"]
    assert job["nodes"]["ev"]["status"] == "skipped"
    assert job["nodes"]["s"]["status"] == "succeeded"  # earlier results are preserved


def test_unencoded_categorical_message():
    r = client.post("/api/datasets", files={"file": ("d.csv", csv_bytes(), "text/csv")}).json()
    wf = {"nodes": [N("d", "data_input", source="upload", dataset_id=r["meta"]["id"]), N("t", "drop_missing", mode="rows"), N("tt", "select_target", target="label", task="classification"),
                    N("s", "train_test_split"), N("m", "clf_random_forest")],
          "edges": [E("d", "dataset", "t", "dataset"), E("t", "dataset", "tt", "dataset"), E("tt", "dataset", "s", "dataset"), E("s", "split", "m", "split")]}
    job = run(wf)
    assert job["nodes"]["m"]["error"]["message"] == "The model received categorical data that has not been encoded. Connect an encoder or configure a preprocessing pipeline."
    assert job["nodes"]["m"]["error"]["technical"]


def test_feature_selection_rf_regression_visualization():
    wf = {"nodes": [N("d", "data_input", builtin="diabetes"), N("t", "select_target", target="target", task="regression"), N("s", "train_test_split"),
                    N("fs", "pp_feature_select", k=4), N("m", "reg_random_forest", n_estimators=30), N("ev", "evaluate"), N("pi", "chart_model", chart="permutation_importance")],
          "edges": [E("d", "dataset", "t", "dataset"), E("t", "dataset", "s", "dataset"), E("s", "split", "fs", "split"), E("fs", "split", "m", "split"),
                    E("m", "trained", "ev", "trained"), E("m", "trained", "pi", "trained")]}
    job = run(wf)
    assert job["status"] == "Completed", {k: v["error"] for k, v in job["nodes"].items() if v["error"]}
    assert len(job["nodes"]["fs"]["preview"]["columns_after"]) == 4
    assert job["nodes"]["ev"]["preview"]["metrics"]["r2"] > 0.2


def test_kmeans_cluster_visualization():
    wf = {"nodes": [N("d", "data_input", builtin="synthetic_clusters", rows=400), N("k", "clustering", algorithm="kmeans", n_clusters=4, label_column="true_cluster"),
                    N("c", "chart_dataset", chart="scatter", x="x1", y="x2", color="cluster")],
          "edges": [E("d", "dataset", "k", "dataset"), E("k", "dataset", "c", "dataset")]}
    job = run(wf)
    assert job["status"] == "Completed"
    m = job["nodes"]["k"]["preview"]["metrics"]
    assert m["adjusted_rand_index"] > 0.8 and m["silhouette"] > 0.3 and sum(job["nodes"]["k"]["preview"]["cluster_sizes"].values()) == 400


@pytest.mark.parametrize("algorithm", ["isolation_forest", "lof", "one_class_svm"])
def test_anomaly_detection_algorithms(algorithm):
    wf = {"nodes": [N("d", "data_input", builtin="synthetic_clusters", rows=180),
                    N("a", "anomaly_detection", algorithm=algorithm, contamination=0.05, n_neighbors=10)],
          "edges": [E("d", "dataset", "a", "dataset")]}
    job = run(wf)
    assert job["status"] == "Completed", {k: v["error"] for k, v in job["nodes"].items() if v["error"]}
    preview = job["nodes"]["a"]["preview"]
    assert preview["kind"] == "anomaly_detection"
    assert preview["anomaly_count"] > 0
    assert "anomaly" in preview["columns"] and "anomaly_score" in preview["columns"]


def test_pca_explained_variance():
    wf = {"nodes": [N("d", "data_input", builtin="wine"), N("p", "pca", n_components=2, keep_columns=["target"])], "edges": [E("d", "dataset", "p", "dataset")]}
    job = run(wf)
    ev = job["nodes"]["p"]["preview"]["cumulative"]
    assert 0.4 < ev[-1] < 1.0


def test_additional_dimensionality_reduction_nodes():
    for node_type in ("tsne", "truncated_svd"):
        wf = {"nodes": [N("d", "data_input", builtin="wine"),
                        N("r", node_type, n_components=2, perplexity=10)],
              "edges": [E("d", "dataset", "r", "dataset")]}
        job = run(wf)
        assert job["status"] == "Completed", {k: v["error"] for k, v in job["nodes"].items() if v["error"]}
        preview = job["nodes"]["r"]["preview"]
        assert preview["kind"] in ("tsne", "truncated_svd")
        assert len(preview["features_used"]) >= 2
        assert len(preview["columns"]) >= 13


@pytest.mark.parametrize("architecture", ["cnn", "lstm", "gru", "transformer"])
def test_pytorch_architectures_train_and_evaluate(architecture):
    wf = {"nodes": [
        N("d", "data_input", builtin="iris"),
        N("t", "select_target", target="target", task="classification"),
        N("s", "train_test_split", test_size=0.25),
        N("m", "pytorch_neural_network", architecture=architecture, epochs=3, hidden=16),
        N("e", "evaluate"),
    ], "edges": [
        E("d", "dataset", "t", "dataset"), E("t", "dataset", "s", "dataset"),
        E("s", "split", "m", "split"), E("m", "trained", "e", "trained"),
    ]}
    job = run(wf)
    assert job["status"] == "Completed", {k: v["error"] for k, v in job["nodes"].items() if v["error"]}
    assert job["nodes"]["m"]["preview"]["algorithm"] == f"PyTorch {architecture.upper()}"
    assert "accuracy" in job["nodes"]["e"]["preview"]["metrics"]


def _bench(n_models=2, seed2=None):
    nodes = [N("d", "data_input", builtin="breast_cancer"), N("t", "select_target", target="target", task="classification"), N("s", "train_test_split", seed=1),
             N("sc", "pp_scaler"), N("m1", "clf_logistic_regression"), N("m2", "clf_random_forest", n_estimators=40), N("e1", "evaluate"), N("e2", "evaluate"), N("cmp", "compare_models")]
    edges = [E("d", "dataset", "t", "dataset"), E("t", "dataset", "s", "dataset"), E("s", "split", "sc", "split"), E("sc", "split", "m1", "split"), E("sc", "split", "m2", "split"),
             E("m1", "trained", "e1", "trained"), E("m2", "trained", "e2", "trained"), E("e1", "metrics", "cmp", "metrics"), E("e2", "metrics", "cmp", "metrics")]
    return {"nodes": nodes, "edges": edges}


def test_compare_models_same_split_is_fair():
    job = run(_bench())
    pv = job["nodes"]["cmp"]["preview"]
    assert job["status"] == "Completed" and len(pv["table"]) == 2 and pv["table"][0]["rank"] == 1
    assert not any("DIFFERENT test sets" in w for w in job["nodes"]["cmp"]["warnings"])


def test_compare_models_different_splits_warns():
    wf = _bench()
    wf["nodes"] += [N("s2", "train_test_split", seed=99), N("m3", "clf_knn"), N("e3", "evaluate")]
    wf["edges"] += [E("t", "dataset", "s2", "dataset"), E("s2", "split", "m3", "split"), E("m3", "trained", "e3", "trained"), E("e3", "metrics", "cmp", "metrics")]
    job = run(wf)
    assert any("DIFFERENT test sets" in w for w in job["nodes"]["cmp"]["warnings"])


def test_changing_config_invalidates_downstream_only():
    wf = _bench()
    run(wf)
    for n in wf["nodes"]:
        if n["id"] == "m2":
            n["params"]["n_estimators"] = 55
    job = run(wf)
    cached = {k: v["cached"] for k, v in job["nodes"].items()}
    assert cached["d"] and cached["s"] and cached["m1"] and cached["e1"]
    assert not cached["m2"] and not cached["e2"] and not cached["cmp"]


def test_partial_execution():
    job = run(_bench(), target="m1")
    st = {k: v["status"] for k, v in job["nodes"].items()}
    assert st["m1"] == "succeeded" and st["m2"] == "not_run" and st["cmp"] == "not_run"


def test_grid_search_uses_training_data_only():
    wf = {"nodes": [N("d", "data_input", builtin="iris"), N("t", "select_target", target="target", task="classification"), N("s", "train_test_split"),
                    N("m", "clf_decision_tree", search="grid", search_space='{"max_depth": [1, 2, 3, 5]}', cv_folds=3), N("ev", "evaluate")],
          "edges": [E("d", "dataset", "t", "dataset"), E("t", "dataset", "s", "dataset"), E("s", "split", "m", "split"), E("m", "trained", "ev", "trained")]}
    job = run(wf)
    s = job["nodes"]["m"]["preview"]["search"]
    assert s["n_trials"] == 4 and "max_depth" in s["best_params"] and s["trials"][0]["rank"] == 1


def test_time_series_split_is_chronological_and_never_shuffled():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"t": np.arange(100), "x": rng.normal(size=100)}); df["y"] = df["x"] * 2
    r = client.post("/api/datasets", files={"file": ("ts.csv", df.sample(frac=1, random_state=1).to_csv(index=False).encode(), "text/csv")}).json()
    wf = {"nodes": [N("d", "data_input", source="upload", dataset_id=r["meta"]["id"]), N("t", "select_target", target="y", task="regression"),
                    N("s", "train_test_split", time_column="t", shuffle=True)], "edges": [E("d", "dataset", "t", "dataset"), E("t", "dataset", "s", "dataset")]}
    job = run(wf)
    assert any("Shuffling a time-ordered" in w for w in job["nodes"]["s"]["warnings"])
    assert job["nodes"]["s"]["preview"]["shuffle"] is False and job["nodes"]["s"]["preview"]["time_ordered"]


def test_target_leak_and_identifier_warnings():
    n = 100
    df = pd.DataFrame({"id": np.arange(n), "x": np.arange(n) * 1.0, "y": np.arange(n) * 2.0 + 1})
    r = client.post("/api/datasets", files={"file": ("l.csv", df.to_csv(index=False).encode(), "text/csv")}).json()
    wf = {"nodes": [N("d", "data_input", source="upload", dataset_id=r["meta"]["id"]), N("t", "select_target", target="y", task="regression")], "edges": [E("d", "dataset", "t", "dataset")]}
    w = run(wf)["nodes"]["t"]["warnings"]
    assert any("identifier" in x for x in w) and any("almost perfectly correlated" in x for x in w)


def test_target_not_in_features():
    wf = {"nodes": [N("d", "data_input", builtin="iris"), N("t", "select_target", target="target", task="classification", features=["target", "sepal length (cm)"])],
          "edges": [E("d", "dataset", "t", "dataset")]}
    pv = run(wf)["nodes"]["t"]["preview"]
    assert "target" not in pv["features"]


def test_async_job_api_progress_and_cancel():
    wf = _bench()
    r = client.post("/api/workflow/run", json={"workflow": {"name": "x", **wf}})
    jid = r.json()["job_id"]
    for _ in range(100):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("Completed", "Failed"):
            break
        time.sleep(0.2)
    assert j["status"] == "Completed" and j["nodes"]["e1"]["preview"]["metrics"]["accuracy"] > 0.9
    assert client.post(f"/api/jobs/{jid}/cancel").status_code == 200


def test_validate_endpoint_explains_invalid_connection():
    wf = {"nodes": [N("d", "data_input"), N("m", "reg_ridge")], "edges": [E("d", "dataset", "m", "split")]}
    res = client.post("/api/workflow/validate", json={"name": "x", **wf}).json()
    assert not res["valid"] and any("cannot connect" in i["message"] for i in res["issues"])


def test_columns_endpoint_and_project_roundtrip():
    r = client.post("/api/datasets", files={"file": ("p.csv", csv_bytes(50, 3), "text/csv")}).json()
    wf = {"nodes": [N("d", "data_input", source="upload", dataset_id=r["meta"]["id"]), N("t", "select_target")], "edges": [E("d", "dataset", "t", "dataset")]}
    cols = client.post("/api/workflow/columns", json={"workflow": {"name": "x", **wf}, "node_id": "t"}).json()["columns"]
    assert [c["name"] for c in cols] == ["x1", "x2", "city", "label"]
    project = {"format": "visual-ml-studio-project", "version": 1, "name": "My Test", "nodes": [{"id": "d", "type": "data_input", "data": {"type": "data_input", "params": {"source": "upload", "dataset_id": "deadbeefdeadbeef"}}}],
               "edges": [], "datasets": [{"id": "deadbeefdeadbeef", "filename": "gone.csv"}]}
    assert client.post("/api/projects", json=project).json()["saved"] == "my-test"
    assert client.get("/api/projects/my-test").json()["name"] == "My Test"
    chk = client.post("/api/projects/check", json=project).json()
    assert chk["missing"][0]["filename"] == "gone.csv"


def test_upload_rejects_bad_files():
    assert client.post("/api/datasets", files={"file": ("x.exe", b"abc", "application/octet-stream")}).status_code == 400
    assert client.post("/api/datasets", files={"file": ("e.csv", b"", "text/csv")}).status_code == 400
