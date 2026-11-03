"""Evaluation + model comparison nodes."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import metrics as M

from ..workflow.spec import NodeError, NodeResult, NodeSpec, Param, Port, register
from ..utils.jsonable import to_jsonable

TR_IN = [Port("trained", "trained", "Trained model")]
MET_OUT = [Port("metrics", "metrics", "Metrics", required=False)]


def _pos_index(classes, pos):
    labels = [str(c) for c in classes]
    if pos in ("", None):
        return len(classes) - 1
    if str(pos) not in labels:
        raise NodeError(f"Positive class '{pos}' is not one of the classes: {labels}.")
    return labels.index(str(pos))


def cls_metrics(y, pred, proba, classes, average, pos):
    out, warns = {}, []
    n_cls = len(classes)
    if average == "binary" and n_cls != 2:
        raise NodeError("'binary' averaging needs exactly two classes. Choose macro or weighted for multiclass problems.")
    pi = _pos_index(classes, pos)
    kw = {"average": average, "zero_division": 0}
    if average == "binary":
        kw["pos_label"] = classes[pi]
    out["accuracy"] = M.accuracy_score(y, pred)
    out["precision"] = M.precision_score(y, pred, **kw)
    out["recall"] = M.recall_score(y, pred, **kw)
    out["f1"] = M.f1_score(y, pred, **kw)
    out["macro_f1"] = M.f1_score(y, pred, average="macro", zero_division=0)
    out["weighted_f1"] = M.f1_score(y, pred, average="weighted", zero_division=0)
    out["balanced_accuracy"] = M.balanced_accuracy_score(y, pred)
    out["mcc"] = M.matthews_corrcoef(y, pred)
    out["cohen_kappa"] = M.cohen_kappa_score(y, pred)
    if proba is not None:
        try:
            if n_cls == 2:
                out["roc_auc"] = M.roc_auc_score(y == classes[pi], proba[:, pi])
                out["pr_auc"] = M.average_precision_score(y == classes[pi], proba[:, pi])
            else:
                out["roc_auc"] = M.roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
            out["log_loss"] = M.log_loss(y, proba, labels=classes)
        except ValueError as e:
            warns.append(f"ROC-AUC/log-loss unavailable: {e}")
    else:
        warns.append("This model does not output probabilities, so ROC-AUC, PR-AUC and log-loss are not shown.")
    return {k: float(v) for k, v in out.items()}, warns


def reg_metrics(y, pred, n_features):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    out, warns = {}, []
    out["mae"] = M.mean_absolute_error(y, pred)
    out["mse"] = M.mean_squared_error(y, pred)
    out["rmse"] = float(np.sqrt(out["mse"]))
    out["r2"] = M.r2_score(y, pred)
    n = len(y)
    if n - n_features - 1 > 0:
        out["adjusted_r2"] = 1 - (1 - out["r2"]) * (n - 1) / (n - n_features - 1)
    out["median_ae"] = M.median_absolute_error(y, pred)
    if np.any(np.abs(y) < 1e-8):
        warns.append("MAPE skipped: some actual targets are zero.")
    else:
        out["mape"] = M.mean_absolute_percentage_error(y, pred)
        if np.any(np.abs(y) < 1e-3 * np.abs(y).mean()):
            warns.append("Some targets are near zero, so MAPE can be misleadingly large.")
    return {k: float(v) for k, v in out.items()}, warns


def insights(tr, test_m, train_m):
    sp, out = tr.split, []
    n_te = len(sp.y_test)
    if n_te < 50:
        out.append(f"Only {n_te} test rows — every metric has wide uncertainty; repeat with other seeds or use cross-validation.")
    if tr.task == "regression":
        r2t, r2s = test_m.get("r2"), train_m.get("r2")
        if r2s is not None and r2t is not None and r2s - r2t > 0.15:
            out.append(f"Likely overfitting: R² is {r2s:.2f} on training data but {r2t:.2f} on test data. Try a simpler model, more regularisation or more data.")
        if r2t is not None and r2t <= 0.05:
            out.append(f"R² = {r2t:.2f}: the model is barely better than always predicting the training mean. The features may not explain the target, or the model is too simple.")
        sd = float(np.std(sp.y_test))
        if sd > 0:
            out.append(f"The typical absolute error (MAE {test_m['mae']:.3g}) is {100*test_m['mae']/sd:.0f}% of the target's standard deviation ({sd:.3g}).")
    else:
        vc = sp.y_test.value_counts(normalize=True)
        base = float(vc.max())
        acc = test_m["accuracy"]
        if acc - base < 0.02:
            out.append(f"Accuracy {acc:.2f} is no better than always predicting the majority class ({base:.2f}).")
        if vc.min() < 0.1 and len(vc) > 1:
            out.append(f"Classes are imbalanced (smallest class = {100*vc.min():.0f}% of test data), so accuracy can look good while the minority class is missed. Check recall, F1 and balanced accuracy.")
        if train_m["accuracy"] - acc > 0.1:
            out.append(f"Likely overfitting: training accuracy {train_m['accuracy']:.2f} vs test accuracy {acc:.2f}.")
    if tr.cv and tr.cv.get("std") is not None and tr.cv["scores"]:
        out.append(f"Cross-validation ({tr.cv['scoring']}) gave {tr.cv['mean']:.3f} ± {tr.cv['std']:.3f} on the training folds; a large ± means the result depends on which rows land in the test set.")
    return out


def run_eval(ctx, p, ins):
    tr = ins["trained"][0]
    sp = tr.split
    nf = len(tr.feature_names)
    if tr.task == "classification":
        m, w = cls_metrics(sp.y_test, tr.y_pred, tr.y_proba, tr.classes, p["average"], p["positive_class"])
        tm, _ = cls_metrics(sp.y_train, tr.y_pred_train, tr.y_proba_train, tr.classes, p["average"], p["positive_class"])
    else:
        m, w = reg_metrics(sp.y_test, tr.y_pred, nf)
        tm, _ = reg_metrics(sp.y_train, tr.y_pred_train, nf)
    ins_ = insights(tr, m, tm)
    payload = {"model": tr.algo_name, "node": ctx.node_id, "task": tr.task, "metrics": m, "train_metrics": tm, "cv": tr.cv,
               "train_time": tr.train_time, "infer_time": tr.infer_time, "params": to_jsonable(tr.kwargs), "fingerprint": sp.fingerprint, "seed": sp.seed,
               "n_train": len(sp.y_train), "n_test": len(sp.y_test)}
    for k, v in m.items():
        ctx.log(f"{k} = {v:.4f}")
    prev = {"kind": "metrics", **to_jsonable(payload), "insights": ins_, "classes": tr.classes}
    return NodeResult({"metrics": payload}, prev, w)


def code_eval(c, nid, p, ins):
    tv, var = ins["trained"][0], c.out(nid, "metrics")
    mt = c.meta[tv]
    sv = mt["split"]
    c.imports.add("from sklearn import metrics as M")
    lines = []
    if mt["task"] == "classification":
        lines += [f"classes = list(pipe_{tv}.classes_)",
                  f"pos_idx = next((i for i, k in enumerate(classes) if str(k) == {p['positive_class']!r}), len(classes) - 1)" if p["positive_class"] else "pos_idx = len(classes) - 1",
                  f"avg = {p['average']!r}",
                  f"kw = dict(average=avg, zero_division=0, **({{'pos_label': classes[pos_idx]}} if avg == 'binary' else {{}}))",
                  f"y_t, y_p = y_test_{sv}, y_pred_{tv}",
                  f"metrics_{var} = {{'accuracy': M.accuracy_score(y_t, y_p), 'precision': M.precision_score(y_t, y_p, **kw), 'recall': M.recall_score(y_t, y_p, **kw),",
                  "    'f1': M.f1_score(y_t, y_p, **kw), 'macro_f1': M.f1_score(y_t, y_p, average='macro', zero_division=0),",
                  "    'weighted_f1': M.f1_score(y_t, y_p, average='weighted', zero_division=0), 'balanced_accuracy': M.balanced_accuracy_score(y_t, y_p),",
                  "    'mcc': M.matthews_corrcoef(y_t, y_p), 'cohen_kappa': M.cohen_kappa_score(y_t, y_p)}",
                  f"if y_proba_{tv} is not None:",
                  "    if len(classes) == 2:",
                  f"        metrics_{var}['roc_auc'] = M.roc_auc_score(y_t == classes[pos_idx], y_proba_{tv}[:, pos_idx])",
                  f"        metrics_{var}['pr_auc'] = M.average_precision_score(y_t == classes[pos_idx], y_proba_{tv}[:, pos_idx])",
                  "    else:",
                  f"        metrics_{var}['roc_auc'] = M.roc_auc_score(y_t, y_proba_{tv}, multi_class='ovr', average='macro', labels=classes)",
                  f"    metrics_{var}['log_loss'] = M.log_loss(y_t, y_proba_{tv}, labels=classes)"]
    else:
        lines += [f"y_t, y_p = y_test_{sv}, y_pred_{tv}",
                  f"metrics_{var} = {{'mae': M.mean_absolute_error(y_t, y_p), 'mse': M.mean_squared_error(y_t, y_p),",
                  "    'rmse': M.root_mean_squared_error(y_t, y_p), 'r2': M.r2_score(y_t, y_p), 'median_ae': M.median_absolute_error(y_t, y_p)}",
                  f"if (y_t != 0).all():\n    metrics_{var}['mape'] = M.mean_absolute_percentage_error(y_t, y_p)"]
    lines += [f"metrics_{var} = {{k: float(v) for k, v in metrics_{var}.items()}}",
              f"pd.Series(metrics_{var}, name={mt['label']!r}).round(4).to_frame()"]
    c.meta[var] = {"model": mt["label"], "task": mt["task"], "trained": tv, "split": sv}
    c.cell(nid, f"Evaluation: {mt['label']}", "Metrics are computed on the **held-out test set**.", "\n".join(lines))
    return {"metrics": var}


register(NodeSpec("evaluate", "Model Evaluation", "Evaluation", "Compute classification or regression metrics on the test set.", TR_IN, MET_OUT,
                  [Param("average", "Averaging (classification)", "select", "macro", "How precision/recall/F1 are averaged over classes.", options=["macro", "weighted", "micro", "binary"]),
                   Param("positive_class", "Positive class (binary)", "string", "", "Class treated as 'positive' for binary metrics. Empty = last class.")],
                  run_eval, code_eval, icon="gauge",
                  doc={"what": "Classification: accuracy, precision, recall, F1, balanced accuracy, MCC, Cohen's kappa, ROC-AUC, PR-AUC, log-loss. Regression: MAE, MSE, RMSE, R², adjusted R², median AE, MAPE.",
                       "how": "Metrics use the held-out test rows. A 'Why did this result happen?' panel explains the numbers using your actual data.",
                       "when_not_to_use": "Never pick a model by accuracy alone on imbalanced data.",
                       "limitations": "Precision vs recall trade off: raise one and the other usually drops; F1 balances them; ROC-AUC measures ranking quality independent of threshold. R² can be negative; RMSE punishes big errors more than MAE."},
                  keywords="metrics accuracy f1 r2 mae rmse roc auc score"))


# ------------------------------------------------------------------ Compare
def run_compare(ctx, p, ins):
    items = ins["metrics"]
    tasks = {m["task"] for m in items}
    if len(tasks) > 1:
        raise NodeError("Cannot compare classification and regression results in one table.")
    task = tasks.pop()
    metric = p["metric"]
    avail = set().union(*[m["metrics"].keys() for m in items])
    if metric == "auto":
        metric = "accuracy" if task == "classification" else "r2"
    if metric not in avail:
        raise NodeError(f"Metric '{metric}' is not available for these models. Available: {sorted(avail)}.")
    lower_better = metric in ("mae", "mse", "rmse", "median_ae", "mape", "log_loss")
    rows = []
    for m in items:
        rows.append({"model": m["model"], "node": m["node"], **{k: v for k, v in m["metrics"].items()}, "cv_mean": (m["cv"] or {}).get("mean"),
                     "train_time_s": m["train_time"], "infer_time_s": m["infer_time"], "fingerprint": m["fingerprint"]})
    df = pd.DataFrame(rows)
    df = df.sort_values(metric, ascending=lower_better, na_position="last").reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    warns = []
    if len({m["fingerprint"] for m in items}) > 1:
        warns.append("These models were evaluated on DIFFERENT test sets (different split seed/size), so the comparison is not strictly fair. Branch from one Train/Test Split node for a fair comparison.")
    if task == "classification":
        warns.append("Ranking by one metric is a choice, not a verdict: check precision/recall trade-offs, class balance and training time before declaring a winner.")
    best = df.iloc[0]["model"]
    ctx.log(f"Ranked by {metric} ({'lower' if lower_better else 'higher'} is better): best = {best}")
    return NodeResult({}, {"kind": "compare", "metric": metric, "lower_is_better": lower_better, "table": to_jsonable(df.to_dict(orient="records")), "best": best, "task": task}, warns)


def code_compare(c, nid, p, ins):
    vs = ins["metrics"]
    task = c.meta[vs[0]]["task"]
    metric = p["metric"] if p["metric"] != "auto" else ("accuracy" if task == "classification" else "r2")
    lower = metric in ("mae", "mse", "rmse", "median_ae", "mape", "log_loss")
    items = ", ".join(f"({c.meta[v]['model']!r}, metrics_{v})" for v in vs)
    code = (f"comparison = pd.DataFrame([{{'model': name, **m}} for name, m in [{items}]])\n"
            f"comparison = comparison.sort_values({metric!r}, ascending={lower}).reset_index(drop=True)\ncomparison")
    c.cell(nid, "Model comparison", f"Ranked by **{metric}** ({'lower' if lower else 'higher'} is better). Models should share the same split for a fair comparison.", code)
    return {}


register(NodeSpec("compare_models", "Compare Models", "Evaluation", "Rank several evaluated models side by side.", [Port("metrics", "metrics", "Metrics (many)", multi=True)], [],
                  [Param("metric", "Rank by", "select", "auto", "'auto' = accuracy (classification) or R² (regression).",
                         options=["auto", "accuracy", "f1", "macro_f1", "balanced_accuracy", "roc_auc", "mcc", "mae", "rmse", "r2"])],
                  run_compare, code_compare, icon="bar-chart",
                  doc={"what": "Collects any number of Model Evaluation nodes into one ranked table with training and inference time.",
                       "how": "Warns when models were evaluated on different test splits (not directly comparable).",
                       "limitations": "The best metric on one test split is not guaranteed to be best in general."},
                  example="One Train/Test Split → three models → three Evaluations → Compare Models", keywords="benchmark ranking leaderboard side by side"))
