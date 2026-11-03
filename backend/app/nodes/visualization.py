"""Dataset and model charts (Plotly JSON built from the actual run data)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn import metrics as M
from sklearn.inspection import permutation_importance

from ..workflow.spec import NodeError, NodeResult, NodeSpec, Param, Port, register
from ..utils.jsonable import fig_json

LAYOUT = dict(template="plotly_white", margin=dict(l=50, r=20, t=50, b=50), height=420)
FIG_OUT = [Port("figure", "figure", "Chart", required=False)]
DATA_CHARTS = ["histogram", "box", "violin", "scatter", "correlation_heatmap", "missing_values", "bar_counts", "ecdf", "density", "pair_plot", "line"]
MODEL_CHARTS = ["actual_vs_predicted", "residuals", "residual_distribution", "confusion_matrix", "roc_curve", "precision_recall_curve",
                "feature_importance", "permutation_importance", "probability_distribution", "class_distribution"]


def _col(df, c, label):
    if not c:
        raise NodeError(f"Choose a column for '{label}'.")
    if c not in df.columns:
        raise NodeError(f"Column '{c}' was not found in the data.")
    return c


def run_data_chart(ctx, p, ins):
    df = ins["dataset"][0].df
    ch, x, y, color = p["chart"], p["x"], p["y"], p["color"] or None
    if color and color not in df.columns:
        raise NodeError(f"Color column '{color}' not found.")
    num = df.select_dtypes(include="number")
    if ch == "histogram":
        fig = px.histogram(df, x=_col(df, x, "X"), color=color, nbins=p["bins"], marginal="box")
    elif ch == "box":
        fig = px.box(df, y=_col(df, y or x, "Y"), x=color)
    elif ch == "violin":
        fig = px.violin(df, y=_col(df, y or x, "Y"), x=color, box=True)
    elif ch == "scatter":
        fig = px.scatter(df, x=_col(df, x, "X"), y=_col(df, y, "Y"), color=color, opacity=0.75)
    elif ch == "line":
        fig = px.line(df.sort_values(_col(df, x, "X")), x=x, y=_col(df, y, "Y"), color=color)
    elif ch == "correlation_heatmap":
        if num.shape[1] < 2:
            raise NodeError("Need at least two numerical columns for a correlation heatmap.")
        corr = num.corr()
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
    elif ch == "missing_values":
        miss = df.isna().mean().mul(100).round(2)
        fig = px.bar(x=miss.index, y=miss.values, labels={"x": "column", "y": "% missing"})
    elif ch == "bar_counts":
        vc = df[_col(df, x, "X")].value_counts(dropna=False).head(30)
        fig = px.bar(x=vc.index.astype(str), y=vc.values, labels={"x": x, "y": "count"})
    elif ch == "ecdf":
        fig = px.ecdf(df, x=_col(df, x, "X"), color=color)
    elif ch == "density":
        fig = px.histogram(df, x=_col(df, x, "X"), color=color, histnorm="probability density", marginal="rug", nbins=p["bins"], opacity=0.7)
    elif ch == "pair_plot":
        cols = list(num.columns)[:6]
        if len(cols) < 2:
            raise NodeError("Need at least two numerical columns.")
        fig = px.scatter_matrix(df, dimensions=cols, color=color, opacity=0.6)
        fig.update_traces(diagonal_visible=False, marker_size=3)
    else:
        raise NodeError(f"Unknown chart '{ch}'.")
    fig.update_layout(**LAYOUT, title=ch.replace("_", " ").title())
    return NodeResult({"figure": True}, {"kind": "figure", "figure": fig_json(fig)}, [])


def code_data_chart(c, nid, p, ins):
    sv = ins["dataset"][0]
    c.imports.add("import plotly.express as px")
    ch, x, y, col = p["chart"], p["x"], p["y"], p["color"] or None
    d = f"df_{sv}"
    code = {"histogram": f"px.histogram({d}, x={x!r}, color={col!r}, nbins={p['bins']}, marginal='box')",
            "box": f"px.box({d}, y={(y or x)!r}, x={col!r})", "violin": f"px.violin({d}, y={(y or x)!r}, x={col!r}, box=True)",
            "scatter": f"px.scatter({d}, x={x!r}, y={y!r}, color={col!r}, opacity=0.75)", "line": f"px.line({d}.sort_values({x!r}), x={x!r}, y={y!r}, color={col!r})",
            "correlation_heatmap": f"px.imshow({d}.select_dtypes('number').corr(), text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale='RdBu_r')",
            "missing_values": f"px.bar({d}.isna().mean().mul(100).round(2), labels={{'value': '% missing', 'index': 'column'}})",
            "bar_counts": f"px.bar({d}[{x!r}].value_counts(dropna=False).head(30))", "ecdf": f"px.ecdf({d}, x={x!r}, color={col!r})",
            "density": f"px.histogram({d}, x={x!r}, color={col!r}, histnorm='probability density', marginal='rug')",
            "pair_plot": f"px.scatter_matrix({d}, dimensions=list({d}.select_dtypes('number').columns[:6]), color={col!r})"}[ch]
    c.cell(nid, f"Chart: {ch.replace('_', ' ')}", "Interactive Plotly chart of the data at this point in the workflow.", f"fig = {code}\nfig.update_layout(title={ch.replace('_', ' ').title()!r})\nfig.show()")
    return {}


register(NodeSpec("chart_dataset", "Dataset Chart", "Visualization", "Histogram, box, violin, scatter, heatmap, missing values, ECDF…", [Port("dataset", "dataset", "Dataset")], FIG_OUT,
                  [Param("chart", "Chart type", "select", "histogram", "", options=DATA_CHARTS),
                   Param("x", "X column", "column", "", "Column on the X axis."), Param("y", "Y column", "column", "", "Column on the Y axis (scatter/box/violin/line)."),
                   Param("color", "Color by (optional)", "column", "", "Use the 'cluster' column to color clusters."),
                   Param("bins", "Bins", "int", 30, "", min=2, max=200, visible_if={"chart": ["histogram", "density"]})],
                  run_data_chart, code_data_chart, icon="chart",
                  doc={"what": "Interactive charts drawn from the actual data flowing into the node. Hover, zoom and download as PNG.", "example": "Clustering → Dataset Chart (scatter, color = cluster)."},
                  keywords="plot chart histogram scatter correlation heatmap missing"))


def _importance(tr):
    est = tr.pipeline.steps[-1][1]
    names = tr.feature_names
    if hasattr(est, "feature_importances_"):
        v = np.asarray(est.feature_importances_)
    elif hasattr(est, "coef_"):
        v = np.abs(np.asarray(est.coef_))
        v = v.mean(axis=0) if v.ndim > 1 else v
        v = np.ravel(v)
    else:
        raise NodeError(f"{tr.algo_name} has no built-in feature importances. Use 'permutation_importance' instead.")
    if len(v) != len(names):
        raise NodeError("Feature names and importances have different lengths; use 'permutation_importance'.")
    return names, v


def run_model_chart(ctx, p, ins):
    tr = ins["trained"][0]
    sp = tr.split
    ch = p["chart"]
    reg_only = ("actual_vs_predicted", "residuals", "residual_distribution")
    cls_only = ("confusion_matrix", "roc_curve", "precision_recall_curve", "probability_distribution", "class_distribution")
    if ch in reg_only and tr.task != "regression":
        raise NodeError(f"'{ch}' is a regression chart, but this is a classification model.")
    if ch in cls_only and tr.task != "classification":
        raise NodeError(f"'{ch}' is a classification chart, but this is a regression model.")
    yt, yp = sp.y_test, tr.y_pred
    warns = []
    if ch == "actual_vs_predicted":
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        fig = go.Figure([go.Scatter(x=yt, y=yp, mode="markers", name="test rows", marker=dict(opacity=.7)),
                         go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect prediction", line=dict(dash="dash"))])
        fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
    elif ch == "residuals":
        res = np.asarray(yt, float) - yp
        fig = px.scatter(x=yp, y=res, labels={"x": "Predicted", "y": "Residual (actual − predicted)"}, opacity=.7)
        fig.add_hline(y=0, line_dash="dash")
    elif ch == "residual_distribution":
        fig = px.histogram(x=np.asarray(yt, float) - yp, nbins=30, labels={"x": "Residual"}, marginal="box")
    elif ch == "confusion_matrix":
        labels = tr.classes
        cm = M.confusion_matrix(yt, yp, labels=labels)
        if p["normalize"]:
            cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig = px.imshow(cm, x=[str(l) for l in labels], y=[str(l) for l in labels], text_auto=".2f" if p["normalize"] else True, color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="share" if p["normalize"] else "count"))
    elif ch in ("roc_curve", "precision_recall_curve", "probability_distribution"):
        if tr.y_proba is None:
            raise NodeError(f"{tr.algo_name} does not provide probabilities (e.g. SVC with probability off, SGD with hinge loss). Enable probabilities or choose another model.")
        classes, pr = tr.classes, tr.y_proba
        fig = go.Figure()
        idxs = [len(classes) - 1] if len(classes) == 2 else range(len(classes))
        for i in idxs:
            yb = (np.asarray(yt) == classes[i]).astype(int)
            if ch == "roc_curve":
                fpr, tpr, _ = M.roc_curve(yb, pr[:, i])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"class {classes[i]} (AUC {M.auc(fpr, tpr):.3f})"))
            elif ch == "precision_recall_curve":
                pre, rec, _ = M.precision_recall_curve(yb, pr[:, i])
                fig.add_trace(go.Scatter(x=rec, y=pre, name=f"class {classes[i]} (AP {M.average_precision_score(yb, pr[:, i]):.3f})"))
            else:
                for pos in (0, 1):
                    fig.add_trace(go.Histogram(x=pr[yb == pos, i], name=f"actual {'in' if pos else 'not in'} class {classes[i]}", opacity=.6, nbinsx=25))
        if ch == "roc_curve":
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="chance"))
            fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        elif ch == "precision_recall_curve":
            fig.update_layout(xaxis_title="Recall", yaxis_title="Precision")
        else:
            fig.update_layout(barmode="overlay", xaxis_title="Predicted probability")
    elif ch == "class_distribution":
        a, b = sp.y_test.value_counts().sort_index(), pd.Series(yp).value_counts().sort_index()
        fig = go.Figure([go.Bar(x=[str(i) for i in a.index], y=a.values, name="actual"), go.Bar(x=[str(i) for i in b.index], y=b.values, name="predicted")])
        fig.update_layout(barmode="group")
    elif ch == "feature_importance":
        names, v = _importance(tr)
        order = np.argsort(v)[::-1][:p["top_n"]][::-1]
        fig = px.bar(x=v[order], y=[names[i] for i in order], orientation="h", labels={"x": "importance", "y": ""})
        warns.append("Built-in importances can be biased toward high-cardinality or correlated features; confirm with permutation importance.")
    elif ch == "permutation_importance":
        ctx.log("Computing permutation importance on the test set…")
        res = permutation_importance(tr.pipeline, sp.X_test, sp.y_test, n_repeats=p["n_repeats"], random_state=sp.seed,
                                     scoring=None)
        names = list(sp.X_test.columns)
        order = np.argsort(res.importances_mean)[::-1][:p["top_n"]][::-1]
        fig = go.Figure(go.Bar(x=res.importances_mean[order], y=[names[i] for i in order], orientation="h",
                               error_x=dict(type="data", array=res.importances_std[order])))
        fig.update_layout(xaxis_title="Drop in score when shuffled (original input columns)")
    else:
        raise NodeError(f"Unknown chart '{ch}'.")
    fig.update_layout(**LAYOUT, title=f"{tr.algo_name}: {ch.replace('_', ' ')}")
    return NodeResult({"figure": True}, {"kind": "figure", "figure": fig_json(fig)}, warns)


def code_model_chart(c, nid, p, ins):
    tv = ins["trained"][0]
    mt = c.meta[tv]
    sv = mt["split"]
    c.imports |= {"import plotly.express as px", "import plotly.graph_objects as go", "from sklearn import metrics as M"}
    ch = p["chart"]
    t = f"yt, yp = y_test_{sv}, y_pred_{tv}\n"
    body = {
        "actual_vs_predicted": t + "fig = px.scatter(x=yt, y=yp, labels={'x': 'Actual', 'y': 'Predicted'}, opacity=.7)\nlo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))\nfig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines', name='perfect', line=dict(dash='dash')))",
        "residuals": t + "fig = px.scatter(x=yp, y=yt - yp, labels={'x': 'Predicted', 'y': 'Residual'}, opacity=.7)\nfig.add_hline(y=0, line_dash='dash')",
        "residual_distribution": t + "fig = px.histogram(x=yt - yp, nbins=30, labels={'x': 'Residual'}, marginal='box')",
        "confusion_matrix": t + f"labels = list(pipe_{tv}.classes_)\ncm = M.confusion_matrix(yt, yp, labels=labels)\n" + ("cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)\n" if p["normalize"] else "")
        + "fig = px.imshow(cm, x=[str(l) for l in labels], y=[str(l) for l in labels], text_auto=True, color_continuous_scale='Blues', labels=dict(x='Predicted', y='Actual'))",
        "roc_curve": f"classes = list(pipe_{tv}.classes_)\nfig = go.Figure()\nfor i in ([len(classes) - 1] if len(classes) == 2 else range(len(classes))):\n    yb = (y_test_{sv} == classes[i]).astype(int)\n    fpr, tpr, _ = M.roc_curve(yb, y_proba_{tv}[:, i])\n    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'class {{classes[i]}} (AUC {{M.auc(fpr, tpr):.3f}})'))\nfig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='chance'))",
        "precision_recall_curve": f"classes = list(pipe_{tv}.classes_)\nfig = go.Figure()\nfor i in ([len(classes) - 1] if len(classes) == 2 else range(len(classes))):\n    yb = (y_test_{sv} == classes[i]).astype(int)\n    pr, rc, _ = M.precision_recall_curve(yb, y_proba_{tv}[:, i])\n    fig.add_trace(go.Scatter(x=rc, y=pr, name=f'class {{classes[i]}}'))",
        "probability_distribution": f"classes = list(pipe_{tv}.classes_)\nfig = go.Figure()\nfor i in range(len(classes)):\n    fig.add_trace(go.Histogram(x=y_proba_{tv}[:, i], name=f'P(class {{classes[i]}})', opacity=.6))\nfig.update_layout(barmode='overlay')",
        "class_distribution": f"fig = go.Figure([go.Bar(x=y_test_{sv}.value_counts().sort_index().index.astype(str), y=y_test_{sv}.value_counts().sort_index().values, name='actual'),\n                go.Bar(x=pd.Series(y_pred_{tv}).value_counts().sort_index().index.astype(str), y=pd.Series(y_pred_{tv}).value_counts().sort_index().values, name='predicted')])",
        "feature_importance": f"est = pipe_{tv}.steps[-1][1]\nnames = list(pipe_{tv}[:-1].transform(X_test_{sv}).columns) if len(pipe_{tv}.steps) > 1 else list(X_test_{sv}.columns)\nimp = est.feature_importances_ if hasattr(est, 'feature_importances_') else np.ravel(np.abs(est.coef_))\norder = np.argsort(imp)[::-1][:{p['top_n']}][::-1]\nfig = px.bar(x=imp[order], y=[names[i] for i in order], orientation='h')",
        "permutation_importance": f"from sklearn.inspection import permutation_importance\nres = permutation_importance(pipe_{tv}, X_test_{sv}, y_test_{sv}, n_repeats={p['n_repeats']}, random_state={mt.get('seed', 42)})\nnames = list(X_test_{sv}.columns)\norder = np.argsort(res.importances_mean)[::-1][:{p['top_n']}][::-1]\nfig = go.Figure(go.Bar(x=res.importances_mean[order], y=[names[i] for i in order], orientation='h', error_x=dict(type='data', array=res.importances_std[order])))",
    }[ch]
    if ch in ("feature_importance", "permutation_importance"):
        c.imports.add("import numpy as np")
    c.cell(nid, f"Chart: {ch.replace('_', ' ')}", f"Computed from the trained model **{mt['label']}**.", body + f"\nfig.update_layout(title={(mt['label'] + ': ' + ch.replace('_', ' '))!r})\nfig.show()")
    return {}


register(NodeSpec("chart_model", "Model Chart", "Visualization", "Actual vs predicted, residuals, confusion matrix, ROC/PR, feature importance.", [Port("trained", "trained", "Trained model")], FIG_OUT,
                  [Param("chart", "Chart type", "select", "actual_vs_predicted", "Regression: actual_vs_predicted, residuals. Classification: confusion_matrix, roc_curve…", options=MODEL_CHARTS),
                   Param("normalize", "Normalise rows", "bool", False, "Confusion matrix as row percentages.", visible_if={"chart": ["confusion_matrix"]}),
                   Param("top_n", "Top N features", "int", 15, "", min=3, max=100, visible_if={"chart": ["feature_importance", "permutation_importance"]}),
                   Param("n_repeats", "Permutation repeats", "int", 5, "", min=1, max=50, advanced=True, visible_if={"chart": ["permutation_importance"]})],
                  run_model_chart, code_model_chart, icon="line-chart",
                  doc={"what": "Charts computed from real predictions on the test set. Hover for values; use the toolbar to zoom or save as PNG.",
                       "limitations": "ROC/PR need predicted probabilities. Built-in feature importance can mislead; permutation importance on held-out data is more trustworthy."},
                  example="Model → Model Chart (actual_vs_predicted)", keywords="plot confusion matrix roc residual importance feature"))
