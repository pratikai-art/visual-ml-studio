"""Clustering and PCA nodes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import metrics as M
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.manifold import TSNE

from ..workflow.payloads import Dataset
from ..workflow.spec import NodeError, NodeResult, NodeSpec, Param, Port, register
from ..utils.jsonable import df_preview, to_jsonable

DS_IN = [Port("dataset", "dataset", "Dataset")]
DS_OUT = [Port("dataset", "dataset", "Dataset")]


def _features(d: Dataset, exclude):
    ex = set(exclude) | {"cluster"} | ({d.target} if d.target else set())
    cols = [c for c in d.df.select_dtypes(include="number").columns if c not in ex]
    if len(cols) < 2:
        raise NodeError("Need at least two numerical feature columns. Encode categorical columns or choose different columns to keep.")
    if d.df[cols].isna().any().any():
        raise NodeError("Numerical features contain missing values. Add 'Drop Missing' (rows) before this node.")
    return cols


def run_cluster(ctx, p, ins):
    d = ins["dataset"][0]
    cols = _features(d, [p["label_column"]] if p["label_column"] else [])
    n = len(d.df)
    warns = []
    if p["algorithm"] in ("agglomerative",) and n > 5000:
        warns.append(f"Agglomerative clustering needs O(n²) memory; {n} rows may be slow.")
    X = d.df[cols].to_numpy(float)
    if p["scale"]:
        X = StandardScaler().fit_transform(X)
    a, k, seed = p["algorithm"], p["n_clusters"], p["seed"]
    est = {"kmeans": lambda: KMeans(n_clusters=k, random_state=seed, n_init=10),
           "minibatch_kmeans": lambda: MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3),
           "dbscan": lambda: DBSCAN(eps=p["eps"], min_samples=p["min_samples"]),
           "agglomerative": lambda: AgglomerativeClustering(n_clusters=k, linkage=p["linkage"]),
           "gmm": lambda: GaussianMixture(n_components=k, random_state=seed)}[a]()
    labels = est.fit_predict(X)
    out = d.df.copy()
    out["cluster"] = labels
    mask = labels != -1
    n_cl = len(set(labels[mask]))
    met = {"n_clusters_found": n_cl, "noise_points": int((~mask).sum())}
    if n_cl >= 2 and mask.sum() > n_cl:
        met["silhouette"] = float(M.silhouette_score(X[mask], labels[mask]))
        met["davies_bouldin"] = float(M.davies_bouldin_score(X[mask], labels[mask]))
        met["calinski_harabasz"] = float(M.calinski_harabasz_score(X[mask], labels[mask]))
    else:
        warns.append("Fewer than 2 clusters were found, so internal metrics cannot be computed. For DBSCAN try a larger eps or smaller min_samples.")
    ref = None
    if p["label_column"]:
        if p["label_column"] not in d.df.columns:
            raise NodeError(f"Reference label column '{p['label_column']}' not found.")
        ref = d.df[p["label_column"]]
        met["adjusted_rand_index"] = float(M.adjusted_rand_score(ref, labels))
        met["normalized_mutual_info"] = float(M.normalized_mutual_info_score(ref, labels))
    ctx.log(f"{a}: {n_cl} clusters, " + ", ".join(f"{k_}={v:.3f}" for k_, v in met.items() if isinstance(v, float)))
    sizes = {str(k_): int(v) for k_, v in pd.Series(labels).value_counts().sort_index().items()}
    prev = {"kind": "clustering", "table": df_preview(out, 100), "columns": [str(c) for c in out.columns], "metrics": met, "cluster_sizes": sizes, "features_used": cols,
            "note": "Silhouette/Davies–Bouldin/Calinski–Harabasz are UNSUPERVISED (no labels needed). ARI/NMI are SUPERVISED comparisons against the reference labels." if ref is not None else "Metrics are unsupervised (no labels needed)."}
    return NodeResult({"dataset": Dataset(out, d.target, d.task)}, to_jsonable(prev), warns)


def code_cluster(c, nid, p, ins):
    sv, var = ins["dataset"][0], c.out(nid, "dataset")
    tgt = c.meta[sv].get("target")
    c.meta[var] = dict(c.meta[sv])
    c.imports |= {"from sklearn.preprocessing import StandardScaler", "from sklearn import metrics as M"}
    a = p["algorithm"]
    ctor = {"kmeans": ("sklearn.cluster", "KMeans", f"n_clusters={p['n_clusters']}, random_state={p['seed']}, n_init=10"),
            "minibatch_kmeans": ("sklearn.cluster", "MiniBatchKMeans", f"n_clusters={p['n_clusters']}, random_state={p['seed']}, n_init=3"),
            "dbscan": ("sklearn.cluster", "DBSCAN", f"eps={p['eps']}, min_samples={p['min_samples']}"),
            "agglomerative": ("sklearn.cluster", "AgglomerativeClustering", f"n_clusters={p['n_clusters']}, linkage={p['linkage']!r}"),
            "gmm": ("sklearn.mixture", "GaussianMixture", f"n_components={p['n_clusters']}, random_state={p['seed']}")}[a]
    c.imports.add(f"from {ctor[0]} import {ctor[1]}")
    excl = [x for x in (p["label_column"], tgt, "cluster") if x]
    code = (f"cols_{var} = [c for c in df_{sv}.select_dtypes('number').columns if c not in {excl!r}]\nX_{var} = df_{sv}[cols_{var}].to_numpy(float)\n"
            + (f"X_{var} = StandardScaler().fit_transform(X_{var})\n" if p["scale"] else "")
            + f"labels_{var} = {ctor[1]}({ctor[2]}).fit_predict(X_{var})\ndf_{var} = df_{sv}.copy()\ndf_{var}['cluster'] = labels_{var}\n"
            f"mask = labels_{var} != -1\nif len(set(labels_{var}[mask])) >= 2:\n    print('Silhouette:', M.silhouette_score(X_{var}[mask], labels_{var}[mask]))\n"
            f"    print('Davies-Bouldin:', M.davies_bouldin_score(X_{var}[mask], labels_{var}[mask]))\n    print('Calinski-Harabasz:', M.calinski_harabasz_score(X_{var}[mask], labels_{var}[mask]))\n")
    if p["label_column"]:
        code += f"print('Adjusted Rand Index (needs reference labels):', M.adjusted_rand_score(df_{sv}[{p['label_column']!r}], labels_{var}))\n"
    code += f"df_{var}['cluster'].value_counts().sort_index()"
    c.cell(nid, f"Clustering: {a}", "Unsupervised clustering; internal metrics need no labels.", code)
    return {"dataset": var}


register(NodeSpec("clustering", "Clustering", "Clustering", "K-Means, MiniBatch K-Means, DBSCAN, Agglomerative or Gaussian Mixture.", DS_IN, DS_OUT,
                  [Param("algorithm", "Algorithm", "select", "kmeans", "", options=["kmeans", "minibatch_kmeans", "dbscan", "agglomerative", "gmm"]),
                   Param("n_clusters", "Clusters (k)", "int", 3, "Number of clusters (not used by DBSCAN).", min=2, max=50, visible_if={"algorithm": ["kmeans", "minibatch_kmeans", "agglomerative", "gmm"]}),
                   Param("eps", "DBSCAN eps", "float", 0.5, "Neighbourhood radius (on scaled data).", min=0.001, max=100, step=0.05, visible_if={"algorithm": ["dbscan"]}),
                   Param("min_samples", "DBSCAN min samples", "int", 5, "", min=1, max=1000, visible_if={"algorithm": ["dbscan"]}),
                   Param("linkage", "Linkage", "select", "ward", "", options=["ward", "complete", "average", "single"], visible_if={"algorithm": ["agglomerative"]}),
                   Param("scale", "Standardise features first", "bool", True, "Distance-based clustering is scale-sensitive."),
                   Param("label_column", "Reference labels (optional)", "column", "", "Excluded from clustering; used only for ARI/NMI."),
                   Param("seed", "Random seed", "int", 42, "", min=0)],
                  run_cluster, code_cluster, icon="shapes",
                  doc={"what": "Groups rows without labels. Adds a 'cluster' column (−1 = DBSCAN noise).", "when_not_to_use": "K-Means assumes round, similarly sized clusters; use DBSCAN for arbitrary shapes.",
                       "limitations": "You must choose k for K-Means; silhouette helps but is not proof. ARI/NMI need true labels and are a different kind of evaluation from silhouette/DB."}, keywords="kmeans dbscan agglomerative gaussian mixture unsupervised"))


def run_anomaly(ctx, p, ins):
    d = ins["dataset"][0]
    cols = _features(d, [])
    X = d.df[cols].to_numpy(float)
    if p["scale"]:
        X = StandardScaler().fit_transform(X)
    algorithm = p["algorithm"]
    if algorithm == "isolation_forest":
        estimator = IsolationForest(
            n_estimators=p["n_estimators"], contamination=p["contamination"],
            random_state=p["seed"], n_jobs=-1,
        )
        labels = estimator.fit_predict(X)
        scores = -estimator.score_samples(X)
    elif algorithm == "lof":
        estimator = LocalOutlierFactor(
            n_neighbors=min(p["n_neighbors"], max(2, len(X) - 1)),
            contamination=p["contamination"],
        )
        labels = estimator.fit_predict(X)
        scores = -estimator.negative_outlier_factor_
    else:
        estimator = OneClassSVM(
            nu=p["contamination"], kernel=p["kernel"], gamma=p["gamma"]
        )
        labels = estimator.fit_predict(X)
        scores = -estimator.decision_function(X)
    out = d.df.copy()
    out["anomaly"] = labels == -1
    out["anomaly_score"] = scores
    count = int((labels == -1).sum())
    warnings = []
    if count == 0:
        warnings.append("No anomalies were detected. Try a higher contamination/nu value or review feature scaling.")
    ctx.log(f"{algorithm}: detected {count} anomalies out of {len(out)} rows.")
    preview = {
        "kind": "anomaly_detection", "table": df_preview(out, 100),
        "columns": [str(c) for c in out.columns], "features_used": cols,
        "algorithm": algorithm, "anomaly_count": count,
        "anomaly_rate": round(count / len(out), 6) if len(out) else 0.0,
    }
    return NodeResult({"dataset": Dataset(out, d.target, d.task)}, to_jsonable(preview), warnings)


def code_anomaly(c, nid, p, ins):
    sv, var = ins["dataset"][0], c.out(nid, "dataset")
    c.meta[var] = dict(c.meta[sv])
    c.imports.add("from sklearn.preprocessing import StandardScaler")
    if p["algorithm"] == "isolation_forest":
        c.imports.add("from sklearn.ensemble import IsolationForest")
        estimator = (
            f"IsolationForest(n_estimators={p['n_estimators']}, "
            f"contamination={p['contamination']!r}, random_state={p['seed']}, n_jobs=-1)"
        )
        score = f"scores_{var} = -model_{var}.score_samples(X_{var})"
    elif p["algorithm"] == "lof":
        c.imports.add("from sklearn.neighbors import LocalOutlierFactor")
        estimator = (
            f"LocalOutlierFactor(n_neighbors={p['n_neighbors']}, "
            f"contamination={p['contamination']!r})"
        )
        score = f"scores_{var} = -model_{var}.negative_outlier_factor_"
    else:
        c.imports.add("from sklearn.svm import OneClassSVM")
        estimator = (
            f"OneClassSVM(nu={p['contamination']!r}, kernel={p['kernel']!r}, "
            f"gamma={p['gamma']!r})"
        )
        score = f"scores_{var} = -model_{var}.decision_function(X_{var})"
    excl = [x for x in (c.meta[sv].get("target"), "anomaly", "anomaly_score") if x]
    code = (
        f"cols_{var} = [c for c in df_{sv}.select_dtypes('number').columns if c not in {excl!r}]\n"
        f"X_{var} = df_{sv}[cols_{var}].to_numpy(float)\n"
        + (f"X_{var} = StandardScaler().fit_transform(X_{var})\n" if p["scale"] else "")
        + f"model_{var} = {estimator}\n"
        + f"labels_{var} = model_{var}.fit_predict(X_{var})\n"
        + score + "\n"
        + f"df_{var} = df_{sv}.copy()\n"
        + f"df_{var}['anomaly'] = labels_{var} == -1\n"
        + f"df_{var}['anomaly_score'] = scores_{var}\n"
        + f"print('Anomalies:', int((labels_{var} == -1).sum()))\n"
    )
    c.cell(nid, f"Anomaly detection: {p['algorithm']}", "Flags unusual rows without target labels.", code)
    return {"dataset": var}


register(NodeSpec(
    "anomaly_detection", "Anomaly Detection", "Anomaly Detection",
    "Find unusual rows with Isolation Forest, Local Outlier Factor or One-Class SVM.",
    DS_IN, DS_OUT,
    [
        Param("algorithm", "Algorithm", "select", "isolation_forest", "", options=["isolation_forest", "lof", "one_class_svm"]),
        Param("contamination", "Expected anomaly fraction", "float", 0.05, "Expected proportion of outliers in the data.", min=0.001, max=0.5, step=0.01),
        Param("n_estimators", "Trees", "int", 100, "Isolation Forest tree count.", min=10, max=1000, visible_if={"algorithm": ["isolation_forest"]}),
        Param("n_neighbors", "Neighbors", "int", 20, "LOF neighborhood size.", min=2, max=500, visible_if={"algorithm": ["lof"]}),
        Param("kernel", "SVM kernel", "select", "rbf", "", options=["linear", "poly", "rbf", "sigmoid"], visible_if={"algorithm": ["one_class_svm"]}),
        Param("gamma", "SVM gamma", "select", "scale", "", options=["scale", "auto"], visible_if={"algorithm": ["one_class_svm"]}),
        Param("scale", "Standardise features first", "bool", True, "Distance and kernel methods are scale-sensitive."),
        Param("seed", "Random seed", "int", 42, "", min=0),
    ],
    run_anomaly, code_anomaly, icon="alert-triangle",
    doc={"what": "Adds boolean anomaly and numeric anomaly_score columns without needing labels.",
         "limitations": "Anomaly detection is sensitive to feature scale, contamination, and irrelevant columns. Validate flagged rows in the original dataset."},
    keywords="outlier anomaly isolation forest local outlier factor lof one class svm novelty",
))


def _reduction_result(d, out, kind, cols, components, explained=None):
    preview = {
        "kind": kind, "table": df_preview(out, 100),
        "columns": [str(c) for c in out.columns], "features_used": cols,
    }
    if explained is not None:
        preview["explained_variance"] = [float(x) for x in explained]
        preview["cumulative"] = [float(x) for x in np.cumsum(explained)]
    return NodeResult({"dataset": Dataset(out, d.target, d.task)}, to_jsonable(preview))


def run_tsne(ctx, p, ins):
    d = ins["dataset"][0]
    cols = _features(d, p["keep_columns"])
    X = d.df[cols].to_numpy(float)
    if p["scale"]:
        X = StandardScaler().fit_transform(X)
    perplexity = min(float(p["perplexity"]), max(2.0, (len(X) - 1) / 3))
    if perplexity != p["perplexity"]:
        ctx.log(f"Adjusted perplexity from {p['perplexity']} to {perplexity:g} for {len(X)} rows.")
    Z = TSNE(
        n_components=p["n_components"], perplexity=perplexity,
        learning_rate="auto", init="pca", random_state=p["seed"],
    ).fit_transform(X)
    out = d.df.copy()
    for i in range(Z.shape[1]):
        out[f"tSNE{i + 1}"] = Z[:, i]
    return _reduction_result(d, out, "tsne", cols, Z)


def code_tsne(c, nid, p, ins):
    sv, var = ins["dataset"][0], c.out(nid, "dataset")
    c.meta[var] = dict(c.meta[sv])
    c.imports |= {"from sklearn.manifold import TSNE", "from sklearn.preprocessing import StandardScaler"}
    excl = [x for x in (c.meta[sv].get("target"), "cluster", "anomaly", "anomaly_score") if x]
    scale = f"X_{var} = StandardScaler().fit_transform(X_{var})\n" if p["scale"] else ""
    names = "[" + ", ".join(f"'tSNE{i + 1}'" for i in range(p["n_components"])) + "]"
    code = (
        f"cols_{var} = [c for c in df_{sv}.select_dtypes('number').columns if c not in {excl!r}]\n"
        f"X_{var} = df_{sv}[cols_{var}].to_numpy(float)\n{scale}"
        f"perplexity_{var} = min({p['perplexity']!r}, max(2.0, (len(X_{var}) - 1) / 3))\n"
        f"Z_{var} = TSNE(n_components={p['n_components']}, perplexity=perplexity_{var}, "
        f"learning_rate='auto', init='pca', random_state={p['seed']}).fit_transform(X_{var})\n"
        f"df_{var} = df_{sv}.copy()\n"
        f"df_{var}[{names}] = Z_{var}\n"
    )
    c.cell(nid, "t-SNE", "Non-linear embedding for visual exploration.", code)
    return {"dataset": var}


register(NodeSpec(
    "tsne", "t-SNE", "Dimensionality Reduction",
    "Non-linear t-SNE embedding for visual exploration.",
    DS_IN, DS_OUT,
    [Param("n_components", "Components", "int", 2, "", min=2, max=3),
     Param("perplexity", "Perplexity", "float", 30.0, "Balances local and global structure; must be smaller than the row count.", min=2, max=1000, step=1),
     Param("scale", "Standardise features first", "bool", True, ""),
     Param("keep_columns", "Columns to keep", "columns", [], "Columns preserved in the output."),
     Param("seed", "Random seed", "int", 42, "", min=0)],
    run_tsne, code_tsne, icon="scatter-chart",
    doc={"what": "Maps numerical rows into two or three dimensions for visual exploration.",
         "limitations": "t-SNE is stochastic, sensitive to perplexity, and should not be used to infer global distances or as a predictive feature transform."},
    keywords="tsne t-sne embedding nonlinear visualization",
))


def run_svd(ctx, p, ins):
    d = ins["dataset"][0]
    cols = _features(d, p["keep_columns"])
    X = d.df[cols].to_numpy(float)
    if p["scale"]:
        X = StandardScaler().fit_transform(X)
    k = min(p["n_components"], len(cols))
    model = TruncatedSVD(n_components=k, random_state=p["seed"])
    Z = model.fit_transform(X)
    out = d.df.copy()
    for i in range(Z.shape[1]):
        out[f"Component{i + 1}"] = Z[:, i]
    return _reduction_result(d, out, "truncated_svd", cols, Z, model.explained_variance_ratio_)


def code_svd(c, nid, p, ins):
    sv, var = ins["dataset"][0], c.out(nid, "dataset")
    c.meta[var] = dict(c.meta[sv])
    c.imports |= {"from sklearn.decomposition import TruncatedSVD", "from sklearn.preprocessing import StandardScaler"}
    excl = [x for x in (c.meta[sv].get("target"), "cluster", "anomaly", "anomaly_score") if x]
    scale = f"X_{var} = StandardScaler().fit_transform(X_{var})\n" if p["scale"] else ""
    names = "[" + ", ".join(f"'Component{i + 1}'" for i in range(p["n_components"])) + "]"
    code = (
        f"cols_{var} = [c for c in df_{sv}.select_dtypes('number').columns if c not in {excl!r}]\n"
        f"X_{var} = df_{sv}[cols_{var}].to_numpy(float)\n{scale}"
        f"model_{var} = TruncatedSVD(n_components=min({p['n_components']}, len(cols_{var})), random_state={p['seed']})\n"
        f"Z_{var} = model_{var}.fit_transform(X_{var})\n"
        f"df_{var} = df_{sv}.copy()\n"
        f"df_{var}[{names}] = Z_{var}\n"
        f"print('Explained variance ratio:', model_{var}.explained_variance_ratio_.round(3))\n"
    )
    c.cell(nid, "Truncated SVD", "Linear dimensionality reduction that also works with sparse-style feature matrices.", code)
    return {"dataset": var}


register(NodeSpec(
    "truncated_svd", "Truncated SVD", "Dimensionality Reduction",
    "Reduce numerical features with truncated singular value decomposition.",
    DS_IN, DS_OUT,
    [Param("n_components", "Components", "int", 2, "", min=2, max=100),
     Param("scale", "Standardise features first", "bool", True, ""),
     Param("keep_columns", "Columns to keep", "columns", [], "Columns preserved in the output."),
     Param("seed", "Random seed", "int", 42, "", min=0)],
    run_svd, code_svd, icon="layers",
    doc={"what": "Projects numerical features into a smaller linear component space.",
         "limitations": "The number of components cannot exceed the number of numerical features; component directions are not target-aware."},
    keywords="svd truncated singular value decomposition dimensionality reduction",
))


def run_pca(ctx, p, ins):
    d = ins["dataset"][0]
    cols = _features(d, p["keep_columns"])
    X = d.df[cols].to_numpy(float)
    if p["scale"]:
        X = StandardScaler().fit_transform(X)
    k = min(p["n_components"], len(cols))
    pca = PCA(n_components=k, random_state=p["seed"])
    Z = pca.fit_transform(X)
    keep = [c for c in p["keep_columns"] if c in d.df.columns] + [c for c in ([d.target] if d.target else []) + ["cluster"] if c in d.df.columns and c not in p["keep_columns"]]
    out = pd.concat([pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(k)], index=d.df.index), d.df[keep]], axis=1)
    ev = pca.explained_variance_ratio_
    ctx.log(f"{k} components explain {100*ev.sum():.1f}% of the variance")
    prev = {"kind": "pca", "table": df_preview(out, 100), "columns": list(out.columns), "explained_variance": [float(x) for x in ev], "cumulative": [float(x) for x in np.cumsum(ev)], "features_used": cols}
    return NodeResult({"dataset": Dataset(out, d.target if d.target in out.columns else None, d.task)}, to_jsonable(prev), [])


def code_pca(c, nid, p, ins):
    sv, var = ins["dataset"][0], c.out(nid, "dataset")
    tgt = c.meta[sv].get("target")
    c.meta[var] = {"target": tgt, "task": c.meta[sv].get("task")}
    c.imports |= {"from sklearn.decomposition import PCA", "from sklearn.preprocessing import StandardScaler"}
    excl = [x for x in (tgt, "cluster", *p["keep_columns"]) if x]
    code = (f"cols_{var} = [c for c in df_{sv}.select_dtypes('number').columns if c not in {excl!r}]\nX_{var} = df_{sv}[cols_{var}].to_numpy(float)\n"
            + (f"X_{var} = StandardScaler().fit_transform(X_{var})\n" if p["scale"] else "")
            + f"pca_{var} = PCA(n_components=min({p['n_components']}, len(cols_{var})), random_state={p['seed']})\nZ_{var} = pca_{var}.fit_transform(X_{var})\n"
            f"keep_{var} = [c for c in {[x for x in (*p['keep_columns'], tgt, 'cluster') if x]!r} if c in df_{sv}.columns]\n"
            f"df_{var} = pd.concat([pd.DataFrame(Z_{var}, columns=[f'PC{{i+1}}' for i in range(Z_{var}.shape[1])], index=df_{sv}.index), df_{sv}[keep_{var}]], axis=1)\n"
            f"print('Explained variance ratio:', pca_{var}.explained_variance_ratio_.round(3))")
    c.cell(nid, "PCA", "Principal Component Analysis projects the data onto directions of maximum variance.", code)
    return {"dataset": var}


register(NodeSpec("pca", "PCA", "Dimensionality Reduction", "Project numerical features onto principal components.", DS_IN, DS_OUT,
                  [Param("n_components", "Components", "int", 2, "Number of principal components.", min=1, max=100),
                   Param("scale", "Standardise first", "bool", True, "PCA is scale-sensitive."),
                   Param("keep_columns", "Columns to keep unchanged", "columns", [], "Labels/cluster columns to carry along."),
                   Param("seed", "Random seed", "int", 42, "", min=0)],
                  run_pca, code_pca, icon="layers",
                  doc={"what": "Linear dimensionality reduction. The preview shows explained variance per component.", "limitations": "Components are linear mixtures — hard to interpret; sensitive to scaling."}, keywords="principal component dimensionality reduction"))
