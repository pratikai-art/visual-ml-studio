"""Machine-learning algorithm nodes: every node trains a real scikit-learn estimator."""
from __future__ import annotations
import importlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold,
                                     TimeSeriesSplit, cross_val_score)

from ..workflow.spec import NodeError, NodeResult, NodeSpec, Param, Port, register
from ..workflow.payloads import Trained
from ..utils.jsonable import df_preview, to_jsonable
from .steps import build_pipeline, step_code

# ---------- parameter helpers ----------
def seed(): return Param("random_state", "Random seed", "int", 42, "Makes results reproducible.", min=0)
def n_est(d=100): return Param("n_estimators", "Number of trees/estimators", "int", d, "More trees = more stable but slower.", min=1, max=2000)
def depth(d=None): return Param("max_depth", "Max depth", "int", d, "Maximum tree depth. Empty = unlimited. Smaller values reduce overfitting.", min=1, max=100, nullable=True)
def msplit(): return Param("min_samples_split", "Min samples to split", "int", 2, "Minimum samples needed to split a node.", min=2, max=1000)
def mleaf(): return Param("min_samples_leaf", "Min samples per leaf", "int", 1, "Minimum samples in a leaf. Larger = smoother model.", min=1, max=1000)
def mfeat(d): return Param("max_features", "Features per split", "select", d, "Features considered at each split.", options=["sqrt", "log2", "all"], map={"all": None})
def cweight(): return Param("class_weight", "Class weight", "select", "none", "'balanced' reweights classes inversely to their frequency (useful for imbalance).", options=["none", "balanced"], map={"none": None})
def lr(d=0.1): return Param("learning_rate", "Learning rate", "float", d, "Shrinks each step. Lower needs more estimators.", min=0.0001, max=10, step=0.01)
def C(d=1.0): return Param("C", "C (regularisation inverse)", "float", d, "Larger C = weaker regularisation (can overfit).", min=0.0001, max=10000, step=0.1)
def alpha(d=1.0): return Param("alpha", "Alpha (regularisation)", "float", d, "Strength of the penalty. Larger = simpler model.", min=0.0, max=1000, step=0.1)
def maxit(d): return Param("max_iter", "Max iterations", "int", d, "Optimizer iteration cap.", min=1, max=100000)
def hidden(): return Param("hidden_layer_sizes", "Hidden layers", "string", "64,32", "Comma-separated layer sizes, e.g. 64,32.", map=lambda v: tuple(int(x) for x in str(v).split(",") if x.strip()))
def activation(): return Param("activation", "Activation", "select", "relu", "Neuron non-linearity.", options=["relu", "tanh", "logistic", "identity"])

DOCS_DEFAULT = {
    "classification": dict(preprocessing="Encode categorical features; impute missing values; scale features for distance/linear/margin-based models.",
                           metrics="Accuracy for balanced classes; F1, balanced accuracy, ROC-AUC/PR-AUC when classes are imbalanced."),
    "regression": dict(preprocessing="Encode categorical features; impute missing values; scale features for distance/linear/margin-based models.",
                       metrics="MAE (interpretable units), RMSE (penalises big errors), R-squared (variance explained)."),
}

# key, task, label, class path, params, fixed kwargs, handles NaN, (what, how, use, avoid, pitfalls)
MODELS = []
def M(key, task, label, path, params, fixed=None, nan=False, doc=("", "", "", "", ""), cx="", sub=None):
    MODELS.append(dict(key=key, task=task, label=label, path=path, params=params, fixed=fixed or {}, nan=nan, doc=doc, cx=cx, sub=sub))

CL, RG = "classification", "regression"
M("clf_logistic_regression", CL, "Logistic Regression", "sklearn.linear_model.LogisticRegression",
  [C(), Param("solver", "Solver", "select", "lbfgs", "Optimisation algorithm.", options=["lbfgs", "liblinear", "saga", "newton-cg"]), maxit(1000), cweight()],
  doc=("A linear model that estimates class probabilities with the logistic (sigmoid/softmax) function.",
       "Learns one weight per feature; the weighted sum is squashed into a probability. Training minimises log-loss with L2 regularisation.",
       "A strong, fast, interpretable baseline for linearly separable problems.",
       "Strongly non-linear boundaries or unscaled features (scale first).",
       "Unscaled features slow convergence (ConvergenceWarning). Coefficients are only comparable after scaling."), cx="O(n·d) per iteration", sub="Linear models")
M("clf_knn", CL, "K-Nearest Neighbors", "sklearn.neighbors.KNeighborsClassifier",
  [Param("n_neighbors", "Neighbors (k)", "int", 5, "Number of neighbours that vote.", min=1, max=500),
   Param("weights", "Weights", "select", "uniform", "'distance' makes closer neighbours count more.", options=["uniform", "distance"]),
   Param("p", "Minkowski p", "int", 2, "1 = Manhattan, 2 = Euclidean.", min=1, max=10)],
  doc=("Classifies a point by majority vote of its k closest training points.", "Stores the training data; at prediction time it measures distances and votes.",
       "Small, low-dimensional datasets with meaningful distances.", "Large datasets (slow predictions) or unscaled/high-dimensional data.",
       "Always scale features; small k overfits, large k underfits."), cx="O(n·d) per prediction", sub="Instance-based")
M("clf_svc", CL, "Support Vector Classifier", "sklearn.svm.SVC",
  [C(), Param("kernel", "Kernel", "select", "rbf", "Shape of the decision boundary.", options=["rbf", "linear", "poly", "sigmoid"]),
   Param("gamma", "Gamma", "select", "scale", "Kernel width for rbf/poly/sigmoid.", options=["scale", "auto"]),
   Param("degree", "Degree", "int", 3, "Polynomial degree (poly kernel only).", min=1, max=10),
   Param("probability", "Probability estimates", "bool", True, "Enables predict_proba (slower: uses internal cross-validation)."), cweight(), Param("random_state", "Random seed", "int", 42, "", min=0)],
  doc=("Finds the maximum-margin boundary between classes, optionally in a kernel-induced feature space.", "Support vectors define the margin; C trades margin width against training errors.",
       "Medium-sized datasets with clear margins.", "Very large datasets (training is roughly quadratic in samples).", "Needs scaled features; slow on >50k rows."), cx="O(n²–n³)", sub="Margin-based")
M("clf_decision_tree", CL, "Decision Tree", "sklearn.tree.DecisionTreeClassifier",
  [Param("criterion", "Criterion", "select", "gini", "Split quality measure.", options=["gini", "entropy", "log_loss"]), depth(), msplit(), mleaf(), cweight(), seed()],
  doc=("A flowchart of yes/no questions on features.", "Greedily picks the split that makes child nodes purest, recursively.",
       "Interpretable models, mixed feature types, little preprocessing.", "When stability matters: single trees change a lot with small data changes.",
       "Unlimited depth memorises the training set (overfitting)."), cx="O(n·d·log n)", sub="Trees")
M("clf_random_forest", CL, "Random Forest", "sklearn.ensemble.RandomForestClassifier",
  [n_est(), depth(), msplit(), mleaf(), mfeat("sqrt"), cweight(), Param("bootstrap", "Bootstrap", "bool", True, "Train each tree on a bootstrap sample."), seed()],
  doc=("An ensemble of decision trees whose votes are averaged.",
       "Each tree is trained on a bootstrap sample (random rows with replacement) and considers a random subset of features at each split. Averaging many decorrelated trees reduces variance compared with one tree.",
       "Strong default for tabular data; little preprocessing; handles non-linearity and interactions.",
       "Extrapolation, very high-dimensional sparse data, or when a tiny, interpretable model is required.",
       "Impurity-based feature importance is biased toward high-cardinality features—use permutation importance to double-check."), cx="O(t·n·d·log n)", sub="Ensembles")
M("clf_extra_trees", CL, "Extra Trees", "sklearn.ensemble.ExtraTreesClassifier",
  [n_est(), depth(), msplit(), mleaf(), mfeat("sqrt"), cweight(), seed()],
  doc=("Like Random Forest but split thresholds are chosen at random.", "Extra randomness further decorrelates the trees and is faster to train.", "Noisy tabular data, fast strong baseline.", "When smooth, precise boundaries are needed.", "Can underperform on few informative features."), sub="Ensembles")
M("clf_gradient_boosting", CL, "Gradient Boosting", "sklearn.ensemble.GradientBoostingClassifier",
  [n_est(), lr(), Param("max_depth", "Max depth", "int", 3, "Depth of each weak tree.", min=1, max=20), Param("subsample", "Subsample", "float", 1.0, "Fraction of rows per tree (<1 = stochastic boosting).", min=0.1, max=1, step=0.05), seed()],
  doc=("Builds trees one after another, each correcting the previous errors.", "Fits each new shallow tree to the gradient of the loss of the current ensemble.", "Accuracy-focused tabular work.", "Tiny data or when training time is critical.", "Too many estimators with high learning rate overfits."), sub="Ensembles")
M("clf_hist_gradient_boosting", CL, "Hist Gradient Boosting", "sklearn.ensemble.HistGradientBoostingClassifier",
  [Param("max_iter", "Iterations", "int", 100, "Boosting rounds.", min=1, max=5000), lr(), depth(), Param("random_state", "Random seed", "int", 42, "", min=0)], nan=True,
  doc=("Fast gradient boosting using histogram-binned features; handles missing values natively.", "Bins features into 255 buckets, then boosts trees.", "Large tabular data, missing values.", "Very small datasets.", "Early stopping may be automatic for n>10k."), sub="Ensembles")
M("clf_adaboost", CL, "AdaBoost", "sklearn.ensemble.AdaBoostClassifier",
  [n_est(50), lr(1.0), seed()], doc=("Boosts weak learners by re-weighting misclassified samples.", "Each round increases weights of hard samples.", "Simple boosting baseline.", "Noisy labels (sensitive to outliers).", ""), sub="Ensembles")
M("clf_gaussian_nb", CL, "Gaussian Naive Bayes", "sklearn.naive_bayes.GaussianNB",
  [Param("var_smoothing", "Variance smoothing", "float", 1e-9, "Portion of the largest variance added for stability.", min=0, max=1, step=1e-9)],
  doc=("Applies Bayes' rule assuming features are independent and normally distributed within each class.", "Estimates per-class means/variances.", "Very fast baseline; small data.", "Strongly correlated features.", "Independence assumption rarely holds; probabilities are over-confident."), sub="Probabilistic")
M("clf_lda", CL, "Linear Discriminant Analysis", "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
  [Param("solver", "Solver", "select", "svd", "", options=["svd", "lsqr", "eigen"])],
  doc=("Finds linear combinations of features that best separate classes, assuming Gaussian classes with shared covariance.", "", "Small data, roughly Gaussian classes.", "Classes with different covariance.", ""), sub="Probabilistic")
M("clf_sgd", CL, "SGD Classifier", "sklearn.linear_model.SGDClassifier",
  [Param("loss", "Loss", "select", "log_loss", "log_loss = logistic regression, hinge = linear SVM (no probabilities).", options=["log_loss", "hinge", "modified_huber"]), Param("alpha", "Alpha", "float", 0.0001, "Regularisation.", min=0, max=10, step=0.0001), maxit(1000), seed()],
  doc=("Linear classifier trained by stochastic gradient descent.", "", "Very large datasets.", "Unscaled features.", "Sensitive to feature scaling and learning-rate schedule."), sub="Linear models")
M("clf_mlp", CL, "MLP Classifier", "sklearn.neural_network.MLPClassifier",
  [hidden(), activation(), Param("alpha", "L2 penalty", "float", 0.0001, "", min=0, max=10, step=0.0001), Param("learning_rate_init", "Initial learning rate", "float", 0.001, "", min=0.00001, max=1, step=0.0001), maxit(300), seed()],
  doc=("A small fully-connected neural network (scikit-learn's implementation, not PyTorch).", "Backpropagation with Adam.", "Non-linear tabular patterns with enough data.", "Tiny data; needs scaled inputs.", "ConvergenceWarning if max_iter is too small."), sub="Neural networks")

M("reg_linear", RG, "Linear Regression", "sklearn.linear_model.LinearRegression",
  [Param("fit_intercept", "Fit intercept", "bool", True, "Learn a bias term.")],
  doc=("Fits a straight line (hyperplane) minimising squared errors.", "Ordinary least squares.", "Baseline, interpretable relationships.", "Strong non-linearity or highly collinear features.", "Outliers pull the fit heavily."), sub="Linear models")
M("reg_ridge", RG, "Ridge Regression", "sklearn.linear_model.Ridge", [alpha()],
  doc=("Linear regression with an L2 penalty that shrinks coefficients.", "", "Collinear features.", "", ""), sub="Linear models")
M("reg_lasso", RG, "Lasso Regression", "sklearn.linear_model.Lasso", [alpha(), maxit(10000)],
  doc=("Linear regression with an L1 penalty that can zero out coefficients.", "", "Sparse feature selection.", "", "Needs scaled features."), sub="Linear models")
M("reg_elastic_net", RG, "Elastic Net", "sklearn.linear_model.ElasticNet",
  [alpha(), Param("l1_ratio", "L1 ratio", "float", 0.5, "0 = ridge, 1 = lasso.", min=0, max=1, step=0.05), maxit(10000)],
  doc=("Mixes L1 and L2 penalties.", "", "Many correlated features.", "", ""), sub="Linear models")
M("reg_decision_tree", RG, "Decision Tree Regressor", "sklearn.tree.DecisionTreeRegressor", [depth(), msplit(), mleaf(), seed()],
  doc=("Piece-wise constant predictions from a tree of feature thresholds.", "", "", "", "Overfits when deep; cannot extrapolate."), sub="Trees")
M("reg_random_forest", RG, "Random Forest Regressor", "sklearn.ensemble.RandomForestRegressor",
  [n_est(), depth(), msplit(), mleaf(), mfeat("all"), Param("bootstrap", "Bootstrap", "bool", True, "Train each tree on a bootstrap sample."), seed()],
  doc=("An ensemble of regression trees whose predictions are averaged.",
       "Bootstrap sampling gives each tree different rows; averaging reduces variance. More trees stabilise the result; max_depth controls overfitting.",
       "Non-linear tabular regression with little preprocessing.", "Extrapolating beyond the training range.",
       "Cannot predict outside the range seen in training; impurity feature importance can mislead."), sub="Ensembles")
M("reg_extra_trees", RG, "Extra Trees Regressor", "sklearn.ensemble.ExtraTreesRegressor", [n_est(), depth(), msplit(), mleaf(), mfeat("all"), seed()],
  doc=("Random Forest variant with random thresholds.", "", "", "", ""), sub="Ensembles")
M("reg_gradient_boosting", RG, "Gradient Boosting Regressor", "sklearn.ensemble.GradientBoostingRegressor",
  [n_est(), lr(), Param("max_depth", "Max depth", "int", 3, "", min=1, max=20), Param("subsample", "Subsample", "float", 1.0, "", min=0.1, max=1, step=0.05), seed()],
  doc=("Sequentially fitted trees that correct residual errors.", "", "", "", ""), sub="Ensembles")
M("reg_hist_gradient_boosting", RG, "Hist Gradient Boosting Regressor", "sklearn.ensemble.HistGradientBoostingRegressor",
  [Param("max_iter", "Iterations", "int", 100, "", min=1, max=5000), lr(), depth(), Param("random_state", "Random seed", "int", 42, "", min=0)], nan=True,
  doc=("Fast histogram-based boosting; handles missing values.", "", "", "", ""), sub="Ensembles")
M("reg_svr", RG, "Support Vector Regression", "sklearn.svm.SVR",
  [C(), Param("kernel", "Kernel", "select", "rbf", "", options=["rbf", "linear", "poly", "sigmoid"]), Param("epsilon", "Epsilon", "float", 0.1, "Width of the no-penalty tube.", min=0, max=100, step=0.05),
   Param("gamma", "Gamma", "select", "scale", "", options=["scale", "auto"])],
  doc=("Fits a function within an epsilon-tube around the data.", "", "Small/medium data.", "Large data; unscaled targets.", "Needs scaled features; target scale affects C/epsilon."), sub="Margin-based")
M("reg_knn", RG, "K-Nearest Neighbors Regressor", "sklearn.neighbors.KNeighborsRegressor",
  [Param("n_neighbors", "Neighbors (k)", "int", 5, "", min=1, max=500), Param("weights", "Weights", "select", "uniform", "", options=["uniform", "distance"])],
  doc=("Averages the targets of the k closest points.", "", "", "Unscaled data.", ""), sub="Instance-based")
M("reg_adaboost", RG, "AdaBoost Regressor", "sklearn.ensemble.AdaBoostRegressor", [n_est(50), lr(1.0), seed()], doc=("Boosting with re-weighted samples.", "", "", "", ""), sub="Ensembles")
M("reg_sgd", RG, "SGD Regressor", "sklearn.linear_model.SGDRegressor",
  [Param("alpha", "Alpha", "float", 0.0001, "", min=0, max=10, step=0.0001), maxit(1000), seed()], doc=("Linear regression via stochastic gradient descent.", "", "Huge datasets.", "Unscaled features.", ""), sub="Linear models")
M("reg_mlp", RG, "MLP Regressor", "sklearn.neural_network.MLPRegressor",
  [hidden(), activation(), Param("alpha", "L2 penalty", "float", 0.0001, "", min=0, max=10, step=0.0001), maxit(500), seed()], doc=("Fully-connected neural network for regression.", "", "", "Small data.", ""), sub="Neural networks")
M("reg_dummy", RG, "Dummy Regressor (baseline)", "sklearn.dummy.DummyRegressor",
  [Param("strategy", "Strategy", "select", "mean", "Constant prediction.", options=["mean", "median"])],
  doc=("Always predicts the training mean/median. Any real model must beat it.", "", "Baseline for comparison.", "", "R² ≈ 0 is the floor."), sub="Baselines")

SCORING = ["auto", "accuracy", "f1_macro", "balanced_accuracy", "roc_auc", "r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]

COMMON = [
    Param("cv_folds", "Cross-validation folds", "int", 0, "0 = off. K-fold on the TRAINING data only; preprocessing is re-fitted inside every fold.", min=0, max=20, advanced=True),
    Param("cv_scoring", "CV scoring", "select", "auto", "Metric used for CV / search. 'auto' = accuracy or R².", options=SCORING, advanced=True),
    Param("search", "Hyperparameter search", "select", "none", "grid / random search runs on the training data only; the test set is never used for tuning.", options=["none", "grid", "random"], advanced=True),
    Param("search_space", "Search space (JSON)", "text", '{}', 'e.g. {"n_estimators": [50, 100, 200], "max_depth": [3, 5, null]}', advanced=True, visible_if={"search": ["grid", "random"]}),
    Param("n_iter", "Random-search trials", "int", 10, "Trial budget for random search.", min=1, max=500, advanced=True, visible_if={"search": ["random"]}),
]


def _cls(path):
    mod, n = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), n)


def make_kwargs(md, p):
    return {pp.name: pp.py(p[pp.name]) for pp in md["params"]} | md["fixed"]


def check_features(Xt, md):
    bad = [c for c in Xt.columns if not pd.api.types.is_numeric_dtype(Xt[c]) and not pd.api.types.is_bool_dtype(Xt[c])]
    if bad:
        raise NodeError("The model received categorical data that has not been encoded. Connect an encoder or configure a preprocessing pipeline.",
                        f"Non-numeric columns reaching the model: {bad}")
    if not md["nan"] and Xt.isna().any().any():
        cols = Xt.columns[Xt.isna().any()].tolist()
        raise NodeError("The model received missing values. Add an Imputer node before the model (or use a Hist Gradient Boosting model, which handles missing values).",
                        f"Columns with NaN: {cols}")


def make_cv(split, folds):
    if split.time_ordered:
        return TimeSeriesSplit(n_splits=folds)
    if split.task == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=split.seed)
    return KFold(n_splits=folds, shuffle=True, random_state=split.seed)


def scoring_for(task, s):
    if s == "auto":
        return "accuracy" if task == "classification" else "r2"
    if task == "classification" and s in ("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"):
        raise NodeError(f"Scoring '{s}' is a regression metric but this is a classification task.")
    if task == "regression" and s in ("accuracy", "f1_macro", "balanced_accuracy", "roc_auc"):
        raise NodeError(f"Scoring '{s}' is a classification metric but this is a regression task.")
    return s


def parse_space(text):
    try:
        space = json.loads(text or "{}")
        assert isinstance(space, dict) and space
        assert all(isinstance(v, list) and v for v in space.values())
    except Exception:
        raise NodeError('Search space must be a JSON object mapping parameter names to non-empty lists, e.g. {"n_estimators": [50, 100]}.')
    return space


def run_model(ctx, p, ins, md):
    split = ins["split"][0]
    if split.task != md["task"]:
        raise NodeError(f"This is a {md['task']} algorithm but the data was set up as a {split.task} task. "
                        f"Choose a {split.task} algorithm or change the task in the 'Select Features & Target' node.")
    kwargs = make_kwargs(md, p)
    est = _cls(md["path"])(**kwargs)
    warns = []
    ctx.log("Fitting preprocessing on training data to verify features…")
    pre = build_pipeline(split.steps, split.task)
    Xt = pre.fit_transform(split.X_train, split.y_train)
    check_features(Xt, md)
    pipe = build_pipeline(split.steps, split.task, est)
    scoring = scoring_for(split.task, p["cv_scoring"])
    search_info = None
    t0 = time.perf_counter()
    if p["search"] != "none":
        space = parse_space(p["search_space"])
        valid = est.get_params()
        for k in space:
            if k not in valid:
                raise NodeError(f"'{k}' is not a parameter of {md['label']}.", f"Valid parameters: {sorted(valid)}")
        folds = max(p["cv_folds"], 3)
        grid = {f"model__{k}": v for k, v in space.items()}
        cv = make_cv(split, folds)
        ctx.log(f"Running {p['search']} search with {folds}-fold CV on training data…")
        if p["search"] == "grid":
            s = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, refit=True)
        else:
            s = RandomizedSearchCV(pipe, grid, n_iter=p["n_iter"], cv=cv, scoring=scoring, refit=True, random_state=split.seed)
        s.fit(split.X_train, split.y_train)
        pipe = s.best_estimator_
        res = pd.DataFrame(s.cv_results_)
        trials = res.sort_values("rank_test_score").head(30)
        search_info = {"best_params": {k.replace("model__", ""): v for k, v in s.best_params_.items()},
                       "best_score": float(s.best_score_), "scoring": scoring, "n_trials": int(len(res)),
                       "trials": [{"params": {k.replace("model__", ""): v for k, v in r["params"].items()},
                                   "mean": float(r["mean_test_score"]), "std": float(r["std_test_score"]), "rank": int(r["rank_test_score"])}
                                  for _, r in trials.iterrows()]}
    else:
        ctx.log(f"Training {md['label']}…")
        pipe.fit(split.X_train, split.y_train)
    train_time = time.perf_counter() - t0
    ctx.check_cancel()
    t1 = time.perf_counter()
    y_pred = pipe.predict(split.X_test)
    infer_time = time.perf_counter() - t1
    y_pred_train = pipe.predict(split.X_train)
    proba = proba_tr = None
    classes = None
    if split.task == "classification":
        classes = [to_jsonable(c) for c in pipe.classes_]
        if hasattr(pipe, "predict_proba"):
            try:
                proba, proba_tr = pipe.predict_proba(split.X_test), pipe.predict_proba(split.X_train)
            except Exception:  # noqa: BLE001
                proba = None
    cv_info = None
    if p["cv_folds"] >= 2 and p["search"] == "none":
        ctx.log(f"Cross-validating ({p['cv_folds']} folds, preprocessing re-fitted per fold)…")
        from sklearn.base import clone
        sc = cross_val_score(clone(build_pipeline(split.steps, split.task, _cls(md["path"])(**kwargs))), split.X_train, split.y_train,
                             cv=make_cv(split, p["cv_folds"]), scoring=scoring)
        cv_info = {"scoring": scoring, "scores": [float(x) for x in sc], "mean": float(sc.mean()), "std": float(sc.std())}
    elif search_info:
        cv_info = {"scoring": scoring, "scores": [], "mean": search_info["best_score"], "std": None, "note": "best score from the hyperparameter search"}
    try:
        names = list(pipe[:-1].transform(split.X_test.head(2)).columns) if len(pipe.steps) > 1 else list(split.X_train.columns)
    except Exception:  # noqa: BLE001
        names = list(Xt.columns)
    ctx.log(f"Trained in {train_time:.2f}s on {len(split.X_train)} rows, {len(names)} features.")
    if len(names) > 500:
        warns.append("More than 500 features reached the model; training may be slow.")
    tr = Trained(pipe, split, md["key"], md["label"], split.task, kwargs, y_pred, proba, y_pred_train, proba_tr, train_time, infer_time, cv_info, search_info, classes, names)
    pred = pd.DataFrame({"row_index": split.X_test.index, "actual": split.y_test.values, "predicted": y_pred})
    if split.task == "regression":
        pred["error"] = pred["predicted"] - pred["actual"]
    elif proba is not None:
        pred["confidence"] = proba.max(axis=1)
    preview = {"kind": "model", "algorithm": md["label"], "task": split.task, "params": to_jsonable(kwargs), "train_time": train_time,
               "infer_time": infer_time, "n_train": len(split.X_train), "n_test": len(split.X_test), "n_features": len(names), "features": names[:200],
               "cv": cv_info, "search": search_info, "predictions": df_preview(pred, 100)}
    return NodeResult({"trained": tr}, preview, warns)


def code_model(c, nid, p, ins, md):
    sv = ins["split"][0]
    meta = c.meta[sv]
    task = meta["task"]
    var = c.out(nid, "trained")
    c.imports.add("from sklearn.pipeline import Pipeline")
    mod, name = md["path"].rsplit(".", 1)
    c.imports.add(f"from {mod} import {name}")
    kwargs = make_kwargs(md, p)
    kws = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    lines = [f"model_{var} = {name}({kws})",
             f"pipe_{var} = Pipeline(steps_{sv} + [('model', model_{var})])"]
    scoring = scoring_for(task, p["cv_scoring"])
    cv_expr = _cv_code(c, meta, max(p["cv_folds"], 3) if p["search"] != "none" else p["cv_folds"])
    if p["search"] != "none":
        space = parse_space(p["search_space"])
        grid = {f"model__{k}": v for k, v in space.items()}
        if p["search"] == "grid":
            c.imports.add("from sklearn.model_selection import GridSearchCV")
            lines.append(f"search_{var} = GridSearchCV(pipe_{var}, {grid!r}, cv={cv_expr}, scoring={scoring!r}, refit=True)")
        else:
            c.imports.add("from sklearn.model_selection import RandomizedSearchCV")
            lines.append(f"search_{var} = RandomizedSearchCV(pipe_{var}, {grid!r}, n_iter={p['n_iter']}, cv={cv_expr}, scoring={scoring!r}, refit=True, random_state={meta['seed']})")
        lines += [f"search_{var}.fit(X_train_{sv}, y_train_{sv})",
                  f"pipe_{var} = search_{var}.best_estimator_",
                  f"print('Best parameters:', search_{var}.best_params_)",
                  f"print('Best CV score:', round(search_{var}.best_score_, 4))"]
    else:
        lines.append(f"pipe_{var}.fit(X_train_{sv}, y_train_{sv})")
        if p["cv_folds"] >= 2:
            c.imports.add("from sklearn.model_selection import cross_val_score")
            c.imports.add("from sklearn.base import clone")
            lines += [f"cv_scores_{var} = cross_val_score(clone(pipe_{var}), X_train_{sv}, y_train_{sv}, cv={cv_expr}, scoring={scoring!r})",
                      f"print('CV {scoring}: %.4f ± %.4f' % (cv_scores_{var}.mean(), cv_scores_{var}.std()))"]
    lines += [f"y_pred_{var} = pipe_{var}.predict(X_test_{sv})"]
    if task == "classification":
        lines += [f"y_proba_{var} = pipe_{var}.predict_proba(X_test_{sv}) if hasattr(pipe_{var}, 'predict_proba') else None"]
    c.meta[var] = {"split": sv, "task": task, "label": md["label"], "key": md["key"], "has_proba": task == "classification", "seed": meta["seed"]}
    c.cell(nid, f"Train: {md['label']}", "Builds a scikit-learn `Pipeline` (preprocessing + model). Preprocessing is fitted on the training data only.", "\n".join(lines))
    return {"trained": var}


def _cv_code(c, meta, folds):
    if meta.get("time_ordered"):
        c.imports.add("from sklearn.model_selection import TimeSeriesSplit")
        return f"TimeSeriesSplit(n_splits={folds})"
    if meta["task"] == "classification":
        c.imports.add("from sklearn.model_selection import StratifiedKFold")
        return f"StratifiedKFold(n_splits={folds}, shuffle=True, random_state={meta['seed']})"
    c.imports.add("from sklearn.model_selection import KFold")
    return f"KFold(n_splits={folds}, shuffle=True, random_state={meta['seed']})"


def _register_all():
    for md in MODELS:
        d = md["doc"]
        base = DOCS_DEFAULT[md["task"]]
        doc = {"what": d[0], "how": d[1] or "See scikit-learn documentation for algorithm details.", "when_to_use": d[2], "when_not_to_use": d[3],
               "limitations": d[4], "preprocessing": base["preprocessing"], "metrics": base["metrics"],
               "complexity": md["cx"] or "Depends on data size and settings.",
               "hyperparameters": "; ".join(f"{x.label}: {x.description}" for x in md["params"] if x.description)}
        doc = {k: v for k, v in doc.items() if v}
        register(NodeSpec(
            type=md["key"], name=md["label"], category="Machine Learning",
            subcategory=("Classification" if md["task"] == CL else "Regression") + (f" · {md['sub']}" if md["sub"] else ""),
            description=d[0] or md["label"], icon="brain",
            inputs=[Port("split", "split", "Train/test data")], outputs=[Port("trained", "trained", "Trained model", required=False)],
            params=md["params"] + COMMON,
            run=lambda ctx, p, ins, md=md: run_model(ctx, p, ins, md),
            codegen=lambda c, nid, p, ins, md=md: code_model(c, nid, p, ins, md),
            doc=doc, example=f"Split → (Imputer → Scaler) → {md['label']} → Evaluation", keywords=f"{md['task']} {md['label']} {md['sub'] or ''} algorithm model"))


_register_all()
