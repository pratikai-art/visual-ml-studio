"""Data input, exploration, cleaning, target selection, splitting and preprocessing nodes."""
from __future__ import annotations
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..datasets import builtin, store
from ..workflow.payloads import Dataset, Split
from ..workflow.spec import NodeError, NodeResult, NodeSpec, Param, Port, register
from ..utils.jsonable import df_preview, to_jsonable
from .steps import build_pipeline, step_code

DS_IN = [Port("dataset", "dataset", "Dataset")]
DS_OUT = [Port("dataset", "dataset", "Dataset")]
SP_IN = [Port("split", "split", "Train/test data")]
SP_OUT = [Port("split", "split", "Train/test data")]


def _ds(ins) -> Dataset:
    return ins["dataset"][0]


def _carry(d: Dataset, df: pd.DataFrame) -> Dataset:
    return Dataset(df, d.target if d.target in df.columns else None, d.task, list(d.notes))


def _summary(df: pd.DataFrame, extra: dict | None = None) -> dict:
    miss = df.isna().sum()
    return {"kind": "table", "table": df_preview(df, 100), "columns": [str(c) for c in df.columns],
            "missing": {str(k): int(v) for k, v in miss.items() if v}, "shape": [int(df.shape[0]), int(df.shape[1])], **(extra or {})}


# ------------------------------------------------------------------ Data Input
def run_input(ctx, p, ins):
    if p["source"] == "builtin":
        key = p["builtin"]
        if key not in builtin.BUILTINS:
            raise NodeError(f"Unknown built-in dataset '{key}'.")
        df = builtin.load_builtin(key, p["rows"], p["seed"])
        info = builtin.BUILTINS[key]
        label = info["label"]
        extra = {"source": info["source"], "license": info["license"], "suggested_target": info["target"], "suggested_task": info["task"]}
    else:
        if not p["dataset_id"]:
            raise NodeError("No dataset selected. Upload a file with 'Upload Dataset' and choose it here, or switch Source to a built-in dataset.")
        try:
            df = store.load_df(p["dataset_id"])
        except store.DatasetError as e:
            raise NodeError(str(e))
        label = store.get_meta(p["dataset_id"])["filename"]
        extra = {}
    warns = []
    if p["sample_rows"] and p["sample_rows"] < len(df):
        df = df.sample(n=p["sample_rows"], random_state=p["seed"]).reset_index(drop=True)
        warns.append(f"Dataset was explicitly sampled to {p['sample_rows']} rows (seed {p['seed']}).")
    ctx.log(f"Loaded '{label}': {df.shape[0]} rows × {df.shape[1]} columns")
    if len(df) > 200_000:
        warns.append("Large dataset: some algorithms may be slow. Consider sampling.")
    return NodeResult({"dataset": Dataset(df)}, _summary(df, {"name": label, **extra}), warns)


def code_input(c, nid, p, ins):
    var = c.out(nid, "dataset")
    c.imports |= {"import pandas as pd", "import numpy as np"}
    if p["source"] == "builtin":
        code = builtin.builtin_code(p["builtin"], p["rows"], p["seed"]).rstrip() + f"\ndf_{var} = df.copy()"
        md = f"Built-in dataset **{builtin.BUILTINS[p['builtin']]['label']}** — source: {builtin.BUILTINS[p['builtin']]['source']}."
    else:
        meta = store.get_meta(p["dataset_id"]) if p["dataset_id"] else {"filename": "dataset.csv", "ext": ".csv"}
        fn, ext = meta["filename"], meta["ext"]
        reader = {".csv": "pd.read_csv(DATASET_PATH)", ".tsv": 'pd.read_csv(DATASET_PATH, sep="\\t")', ".xlsx": "pd.read_excel(DATASET_PATH)",
                  ".parquet": "pd.read_parquet(DATASET_PATH)", ".json": "pd.read_json(DATASET_PATH)", ".jsonl": "pd.read_json(DATASET_PATH, lines=True)"}[ext]
        if ext == ".csv" and meta.get("delimiter") not in (None, ","):
            reader = f"pd.read_csv(DATASET_PATH, sep={meta['delimiter']!r})"
        code = (f"# Place your dataset next to this notebook (or change the path).\n# Google Colab: from google.colab import files; files.upload()\n"
                f"from pathlib import Path\nDATASET_PATH = Path({fn!r})\nif not DATASET_PATH.exists():\n    raise FileNotFoundError(f'Put {{DATASET_PATH}} next to this notebook.')\n"
                f"df_{var} = {reader}")
        md = f"Loads **{fn}**. The original upload path from the app is intentionally not embedded."
    if p["sample_rows"]:
        code += f"\ndf_{var} = df_{var}.sample(n=min({p['sample_rows']}, len(df_{var})), random_state={p['seed']}).reset_index(drop=True)"
    code += f"\nprint(df_{var}.shape)\ndf_{var}.head()"
    c.meta[var] = {"target": None, "task": None}
    c.cell(nid, "Dataset loading", md, code)
    return {"dataset": var}


register(NodeSpec(
    "data_input", "Dataset Input", "Data Input", "Loads an uploaded file or a built-in educational dataset.", [], DS_OUT,
    [Param("source", "Source", "select", "builtin", "Where the data comes from.", options=["builtin", "upload"]),
     Param("builtin", "Built-in dataset", "select", "student_scores", "Educational datasets with documented sources.", options=list(builtin.BUILTINS), visible_if={"source": ["builtin"]}),
     Param("dataset_id", "Uploaded dataset", "string", "", "Choose an uploaded dataset.", visible_if={"source": ["upload"]}),
     Param("rows", "Rows (synthetic only)", "int", 300, "Number of generated rows.", min=20, max=200000, visible_if={"source": ["builtin"]}),
     Param("seed", "Random seed", "int", 42, "Used for generation and sampling.", min=0),
     Param("sample_rows", "Sample rows (0 = all)", "int", 0, "Explicit, reproducible sampling for large datasets.", min=0, max=10_000_000, advanced=True)],
    run_input, code_input, icon="database",
    doc={"what": "Entry point of every workflow. Reads a CSV/TSV/Excel/Parquet/JSON file or a built-in dataset into a pandas DataFrame.",
         "how": "Files are stored by content hash, so re-uploading the same file reconnects saved projects automatically.",
         "when_to_use": "Always the first node.", "limitations": "Large files are limited by the upload limit and RAM."},
    example="Dataset Input → Dataset Explorer", keywords="csv excel parquet upload load data builtin iris titanic synthetic"))


# ------------------------------------------------------------------ Exploration
def run_explorer(ctx, p, ins):
    d = _ds(ins)
    prof = store.profile_df(d.df, 100)
    ctx.log(f"Profiled {prof['rows']} rows, {prof['cols']} columns, {prof['duplicates']} duplicate rows.")
    warnings = [w for w in prof["warnings"] if "duplicate rows" not in w.lower()]
    prof["warnings"] = warnings
    return NodeResult({"dataset": d}, {**_summary(d.df), "kind": "profile", "profile": prof}, warnings[:10])


def code_explorer(c, nid, p, ins):
    sv = ins["dataset"][0]
    var = c.out(nid, "dataset")
    c.meta[var] = dict(c.meta[sv])
    c.cell(nid, "Exploratory data analysis & data quality",
           "Shape, types, missing values, duplicates and descriptive statistics.",
           f"df_{var} = df_{sv}\nprint(df_{var}.shape)\ndisplay(df_{var}.dtypes.to_frame('dtype').join(df_{var}.isna().sum().rename('missing')))\n"
           f"print('Duplicate rows:', int(df_{var}.duplicated().sum()))\ndf_{var}.describe(include='all').T")
    return {"dataset": var}


register(NodeSpec("data_explorer", "Dataset Explorer", "Data Exploration", "Profiles the dataset: types, missing values, duplicates, statistics.", DS_IN, DS_OUT, [],
                  run_explorer, code_explorer, icon="search",
                  doc={"what": "Passes the data through unchanged and computes a profile (types, missing values, duplicates, descriptive statistics, warnings).",
                       "when_to_use": "Right after loading data and again after cleaning, to see what changed."},
                  keywords="eda profile statistics describe quality"))


def run_missing(ctx, p, ins):
    d = _ds(ins)
    n = len(d.df)
    miss = d.df.isna().sum()
    tbl = [{"column": str(c), "missing": int(m), "percent": round(100 * m / n, 2) if n else 0} for c, m in miss.items()]
    rows_any = int(d.df.isna().any(axis=1).sum())
    ctx.log(f"{int(miss.sum())} missing cells; {rows_any} rows contain at least one gap.")
    return NodeResult({"dataset": d}, {**_summary(d.df), "kind": "missing", "missing_table": tbl, "rows_with_missing": rows_any},
                      [f"{rows_any} rows contain missing values."] if rows_any else [])


def code_missing(c, nid, p, ins):
    sv = ins["dataset"][0]
    var = c.out(nid, "dataset")
    c.meta[var] = dict(c.meta[sv])
    c.cell(nid, "Missing-value analysis", "Counts missing values per column. Imputation happens later, **after** the train/test split, to avoid leakage.",
           f"df_{var} = df_{sv}\n(df_{var}.isna().sum().to_frame('missing').assign(percent=lambda t: (100 * t['missing'] / len(df_{var})).round(2)))")
    return {"dataset": var}


register(NodeSpec("missing_analysis", "Missing-Value Analysis", "Data Exploration", "Reports missing values per column (does not modify data).", DS_IN, DS_OUT, [],
                  run_missing, code_missing, icon="alert",
                  doc={"what": "Shows how many values are missing in each column.", "when_to_use": "Before deciding between dropping rows and imputing.",
                       "limitations": "Detection only — use an Imputer node (after the split) or Drop Missing node to act."}))


# ------------------------------------------------------------------ Cleaning
def run_dups(ctx, p, ins):
    d = _ds(ins)
    sub = p["subset"] or None
    for c in p["subset"]:
        if c not in d.df.columns:
            raise NodeError(f"Column '{c}' does not exist.")
    mask = d.df.duplicated(subset=sub, keep="first" if p["keep"] == "first" else "last" if p["keep"] == "last" else False)
    removed = d.df[mask]
    out = d.df if p["mode"] == "detect" else d.df[~mask]
    ctx.log(f"{int(mask.sum())} duplicate rows {'found' if p['mode']=='detect' else 'removed'}.")
    return NodeResult({"dataset": _carry(d, out)}, {**_summary(out, {"duplicates_preview": df_preview(removed, 20), "n_duplicates": int(mask.sum()),
                                                                     "duplicates_removed": p["mode"] == "remove"})}, [])


def code_dups(c, nid, p, ins):
    sv, var = ins["dataset"][0], c.out(nid, "dataset")
    c.meta[var] = dict(c.meta[sv])
    keep = {"first": "'first'", "last": "'last'", "none": "False"}[p["keep"]]
    sub = repr(p["subset"]) if p["subset"] else "None"
    body = (f"mask = df_{sv}.duplicated(subset={sub}, keep={keep})\nprint('Duplicates:', int(mask.sum()))\n"
            + (f"df_{var} = df_{sv}[~mask]" if p["mode"] == "remove" else f"df_{var} = df_{sv}"))
    c.cell(nid, "Duplicate handling", "Detects and optionally removes duplicate rows.", body)
    return {"dataset": var}


register(NodeSpec("drop_duplicates", "Duplicate Rows", "Data Cleaning", "Detect or remove duplicate rows.", DS_IN, DS_OUT,
                  [Param("mode", "Mode", "select", "remove", "'detect' only reports; 'remove' deletes duplicates.", options=["detect", "remove"]),
                   Param("subset", "Identify by columns", "columns", [], "Empty = all columns."),
                   Param("keep", "Keep", "select", "first", "Which duplicate to keep.", options=["first", "last", "none"])],
                  run_dups, code_dups, icon="copy", doc={"what": "Finds rows that repeat and (optionally) drops them. Preview of the duplicates is shown before you rely on removal."}))


def run_dropcols(ctx, p, ins):
    d = _ds(ins)
    cols = list(p["columns"])
    missing = [c for c in cols if c not in d.df.columns]
    if missing:
        raise NodeError(f"Columns not found: {missing}")
    df = d.df.drop(columns=cols)
    const = []
    if p["drop_constant"]:
        const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1 and c != d.target]
        df = df.drop(columns=const)
    if d.target in cols:
        raise NodeError(f"'{d.target}' is the target column; it cannot be dropped here.")
    ctx.log(f"Dropped {len(cols) + len(const)} columns: {cols + const}")
    return NodeResult({"dataset": _carry(d, df)}, _summary(df, {"dropped": cols + const}), [f"Dropped constant columns: {const}"] if const else [])


def code_dropcols(c, nid, p, ins):
    sv, var = ins["dataset"][0], c.out(nid, "dataset")
    c.meta[var] = dict(c.meta[sv])
    lines = [f"df_{var} = df_{sv}.drop(columns={list(p['columns'])!r})"]
    if p["drop_constant"]:
        lines.append(f"df_{var} = df_{var}.loc[:, df_{var}.nunique(dropna=False) > 1]")
    c.cell(nid, "Drop columns", "Removes the listed columns.", "\n".join(lines))
    return {"dataset": var}


register(NodeSpec("drop_columns", "Drop Columns", "Data Cleaning", "Remove identifier or irrelevant columns.", DS_IN, DS_OUT,
                  [Param("columns", "Columns to drop", "columns", [], "Searchable column selector."),
                   Param("drop_constant", "Also drop constant columns", "bool", False, "Columns with a single value carry no information.")],
                  run_dropcols, code_dropcols, icon="trash", doc={"what": "Drops chosen columns (e.g. IDs). The preview lists exactly what was removed."}))


def run_dropmiss(ctx, p, ins):
    d = _ds(ins)
    df = d.df
    before = df.shape
    if p["mode"] == "rows":
        sub = p["subset"] or None
        df = df.dropna(subset=sub)
    else:
        keep = df.columns[df.isna().mean() <= p["max_missing_fraction"]]
        dropped = [c for c in df.columns if c not in keep]
        if d.target in dropped:
            raise NodeError(f"The target column '{d.target}' exceeds the missing threshold; raise the threshold.")
        df = df[keep]
    ctx.log(f"Shape {before} → {df.shape}")
    w = [f"Removed {before[0]-df.shape[0]} rows and {before[1]-df.shape[1]} columns."] if before != df.shape else []
    if len(df) == 0:
        raise NodeError("All rows were removed. Use imputation instead of dropping rows.")
    return NodeResult({"dataset": _carry(d, df)}, _summary(df), w)


def code_dropmiss(c, nid, p, ins):
    sv, var = ins["dataset"][0], c.out(nid, "dataset")
    c.meta[var] = dict(c.meta[sv])
    if p["mode"] == "rows":
        code = f"df_{var} = df_{sv}.dropna(subset={p['subset'] or None!r})"
    else:
        code = f"df_{var} = df_{sv}.loc[:, df_{sv}.isna().mean() <= {p['max_missing_fraction']}]"
    c.cell(nid, "Drop missing values", "Removes rows (or columns) with too many missing values.", code + f"\nprint(df_{sv}.shape, '->', df_{var}.shape)")
    return {"dataset": var}


register(NodeSpec("drop_missing", "Drop Missing", "Data Cleaning", "Remove rows with missing values or columns that are mostly empty.", DS_IN, DS_OUT,
                  [Param("mode", "Remove", "select", "rows", "Rows with gaps, or columns with too many gaps.", options=["rows", "columns"]),
                   Param("subset", "Only check columns", "columns", [], "Empty = all columns.", visible_if={"mode": ["rows"]}),
                   Param("max_missing_fraction", "Max missing fraction", "float", 0.5, "Columns above this are dropped.", min=0, max=1, step=0.05, visible_if={"mode": ["columns"]})],
                  run_dropmiss, code_dropmiss, icon="eraser",
                  doc={"what": "Deletes data with gaps. Never silent: the log reports how many rows/columns went.", "when_not_to_use": "When many rows have gaps—you lose data; prefer imputation after the split."}))


def _num_cols(d, cols):
    use = cols or [c for c in d.df.select_dtypes(include="number").columns if c != d.target]
    for c in use:
        if c not in d.df.columns:
            raise NodeError(f"Column '{c}' does not exist.")
        if not pd.api.types.is_numeric_dtype(d.df[c]):
            raise NodeError(f"Column '{c}' is not numerical.")
    return use


def run_outliers(ctx, p, ins):
    d = _ds(ins)
    cols = _num_cols(d, p["columns"])
    df = d.df.copy()
    flags = pd.DataFrame(False, index=df.index, columns=cols)
    bounds = {}
    for c in cols:
        s = df[c]
        if p["method"] == "zscore":
            m, sd = s.mean(), s.std()
            lo, hi = m - p["threshold"] * sd, m + p["threshold"] * sd
        else:
            q1, q3 = s.quantile(.25), s.quantile(.75)
            lo, hi = q1 - p["threshold"] * (q3 - q1), q3 + p["threshold"] * (q3 - q1)
        flags[c] = (s < lo) | (s > hi)
        bounds[c] = (float(lo), float(hi))
    n_out = int(flags.any(axis=1).sum())
    warns = []
    if p["action"] == "remove":
        df = df[~flags.any(axis=1)]
    elif p["action"] in ("clip", "winsorize"):
        for c in cols:
            df[c] = df[c].clip(*bounds[c])
    if p["action"] != "detect":
        warns.append("Bounds were computed on the full dataset before the train/test split; this can leak a little information. Use for exploration or accept the small leakage.")
    ctx.log(f"{n_out} rows contain outliers ({p['method']}, threshold {p['threshold']}); action = {p['action']}.")
    return NodeResult({"dataset": _carry(d, df)}, _summary(df, {"outlier_rows": n_out, "bounds": bounds}), warns)


def code_outliers(c, nid, p, ins):
    sv, var = ins["dataset"][0], c.out(nid, "dataset")
    c.meta[var] = dict(c.meta[sv])
    tgt = c.meta[sv].get("target")
    cols = repr(p["columns"]) if p["columns"] else f"[c for c in df_{sv}.select_dtypes('number').columns if c != {tgt!r}]"
    calc = (f"mu, sd = s.mean(), s.std()\n    lo, hi = mu - {p['threshold']} * sd, mu + {p['threshold']} * sd" if p["method"] == "zscore"
            else f"q1, q3 = s.quantile(.25), s.quantile(.75)\n    lo, hi = q1 - {p['threshold']} * (q3 - q1), q3 + {p['threshold']} * (q3 - q1)")
    code = (f"df_{var} = df_{sv}.copy()\ncols = {cols}\nflags = pd.DataFrame(False, index=df_{var}.index, columns=cols)\nbounds = {{}}\n"
            f"for col in cols:\n    s = df_{var}[col]\n    {calc}\n    flags[col] = (s < lo) | (s > hi)\n    bounds[col] = (lo, hi)\n"
            f"print('Rows with outliers:', int(flags.any(axis=1).sum()))\n")
    if p["action"] == "remove":
        code += f"df_{var} = df_{var}[~flags.any(axis=1)]"
    elif p["action"] in ("clip", "winsorize"):
        code += f"for col in cols:\n    df_{var}[col] = df_{var}[col].clip(*bounds[col])"
    c.cell(nid, "Outlier handling", f"Method: {p['method']} (threshold {p['threshold']}); action: {p['action']}.", code)
    return {"dataset": var}


register(NodeSpec("outliers", "Outliers", "Data Cleaning", "Detect outliers (Z-score/IQR) and optionally clip or remove them.", DS_IN, DS_OUT,
                  [Param("method", "Detection", "select", "iqr", "Z-score assumes roughly normal data; IQR is robust.", options=["iqr", "zscore"]),
                   Param("threshold", "Threshold", "float", 1.5, "IQR multiplier (1.5 typical) or number of std-devs (3 typical).", min=0.1, max=10, step=0.1),
                   Param("action", "Action", "select", "detect", "Detecting is separate from modifying.", options=["detect", "clip", "winsorize", "remove"]),
                   Param("columns", "Columns", "columns", [], "Empty = all numerical non-target columns.")],
                  run_outliers, code_outliers, icon="activity",
                  doc={"what": "Finds values far outside the typical range and optionally clips/removes them.", "limitations": "Bounds use all rows (pre-split) – small leakage; clipping hides genuine extremes."}))


# ------------------------------------------------------------------ Feature / target selection
def run_target(ctx, p, ins):
    d = _ds(ins)
    df = d.df
    t = p["target"]
    if not t:
        raise NodeError("No target column chosen. Pick the column you want to predict.")
    if t not in df.columns:
        raise NodeError(f"Target column '{t}' was not found. Available: {list(df.columns)[:15]}")
    feats = [c for c in (p["features"] or [c for c in df.columns if c != t]) if c != t]
    miss = [c for c in feats if c not in df.columns]
    if miss:
        raise NodeError(f"Selected features not found: {miss}")
    if not feats:
        raise NodeError("No feature columns selected.")
    if df[t].isna().any():
        raise NodeError(f"The target '{t}' has {int(df[t].isna().sum())} missing values. Drop those rows first (Drop Missing → rows, subset = target).")
    task = p["task"]
    if task == "auto":
        y = df[t]
        task = "classification" if (not pd.api.types.is_numeric_dtype(y) or y.nunique() <= min(10, 0.05 * len(y) + 2)) else "regression"
    warns = []
    n = len(df)
    for c in feats:
        s = df[c]
        if n > 20 and s.nunique() / n > 0.98 and (pd.api.types.is_integer_dtype(s) or not pd.api.types.is_numeric_dtype(s)):
            warns.append(f"'{c}' looks like an identifier (almost every value is unique). Consider excluding it.")
        if task == "regression" and pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(df[t]):
            r = s.corr(df[t])
            if pd.notna(r) and abs(r) > 0.98:
                warns.append(f"'{c}' is almost perfectly correlated with the target (r={r:.3f}); it may leak the answer.")
        if task == "classification" and pd.api.types.is_numeric_dtype(s) and s.nunique() > 1:
            if s.groupby(df[t]).nunique().max() == 1 and df[t].nunique() > 1:
                warns.append(f"'{c}' determines the target perfectly; check for target leakage.")
    if task == "classification" and df[t].nunique() > 50:
        warns.append(f"The target has {df[t].nunique()} distinct values — this looks like regression, not classification.")
    out = df[feats + [t]]
    excluded = [c for c in df.columns if c not in feats and c != t]
    ctx.log(f"Target '{t}' ({task}); {len(feats)} features; excluded: {excluded}")
    res = _summary(out, {"excluded": excluded, "task": task, "target": t, "features": feats})
    if task == "classification":
        res["class_distribution"] = {str(k): int(v) for k, v in df[t].value_counts().items()}
    return NodeResult({"dataset": Dataset(out, t, task, [])}, res, warns)


def code_target(c, nid, p, ins):
    sv, var = ins["dataset"][0], c.out(nid, "dataset")
    t = p["target"]
    task = p["task"]
    feats = p["features"]
    c.meta[var] = {"target": t, "task": task if task != "auto" else c.infer_task(sv, t), "features": feats}
    flist = repr(feats) if feats else f"[c for c in df_{sv}.columns if c != {t!r}]"
    code = (f"TARGET = {t!r}\nFEATURES = {flist}\nTASK = {c.meta[var]['task']!r}  # classification | regression\n"
            f"df_{var} = df_{sv}[FEATURES + [TARGET]]\nprint('Features:', FEATURES)")
    c.cell(nid, "Feature and target selection", f"Predicting **{t}** ({c.meta[var]['task']}). Excluded columns are listed explicitly; the target is never part of the feature matrix.", code)
    return {"dataset": var}


register(NodeSpec("select_target", "Select Features & Target", "Feature Selection", "Choose the target to predict, the task type and which columns are features.", DS_IN, DS_OUT,
                  [Param("target", "Target column", "column", "", "The column the model learns to predict."),
                   Param("task", "Task", "select", "auto", "Be explicit: classification predicts categories, regression predicts numbers.", options=["auto", "classification", "regression"]),
                   Param("features", "Feature columns", "columns", [], "Empty = every column except the target. Searchable selector.")],
                  run_target, code_target, icon="target",
                  doc={"what": "Marks the target column and keeps only chosen features, so the target can never leak into the feature matrix.",
                       "how": "Warns about identifier-like and suspiciously predictive columns, and lists exactly which columns were excluded.",
                       "when_to_use": "Before the Train/Test Split in every supervised workflow."},
                  keywords="select features target label y columns identifier leakage"))


# ------------------------------------------------------------------ Splitting
def run_split(ctx, p, ins):
    d = _ds(ins)
    if not d.target:
        raise NodeError("No target defined. Insert a 'Select Features & Target' node before the split.")
    df = d.df
    warns = []
    time_col = p["time_column"]
    ordered = bool(time_col)
    if time_col:
        if time_col not in df.columns:
            raise NodeError(f"Time column '{time_col}' not found.")
        df = df.sort_values(time_col, kind="stable")
        if p["shuffle"]:
            warns.append("Shuffling a time-ordered dataset leaks future information into training. Shuffle was disabled.")
        shuffle = False
    else:
        shuffle = p["shuffle"]
    X, y = df.drop(columns=[d.target]), df[d.target]
    if time_col and time_col in X.columns and not pd.api.types.is_numeric_dtype(X[time_col]):
        warns.append(f"Time column '{time_col}' is kept as a feature but is not numeric.")
    strat = y if (p["stratify"] and d.task == "classification" and shuffle) else None
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=p["test_size"], random_state=p["seed"], shuffle=shuffle, stratify=strat)
    except ValueError as e:
        raise NodeError("The data could not be split. If you use stratification, every class needs at least 2 rows.", str(e))
    if len(Xte) < 20:
        warns.append(f"Only {len(Xte)} test rows: metrics will be noisy.")
    fp = hashlib.sha1((",".join(map(str, Xte.index)) + str(p["seed"])).encode()).hexdigest()[:10]
    sp = Split(Xtr, Xte, ytr, yte, d.task or "regression", d.target, p["seed"], ordered, [], fp)
    prev = {"kind": "split", "n_train": len(Xtr), "n_test": len(Xte), "test_size": p["test_size"], "seed": p["seed"], "shuffle": shuffle,
            "stratified": strat is not None, "fingerprint": fp, "time_ordered": ordered, "task": sp.task,
            "train_preview": df_preview(Xtr, 20), "columns": [str(c) for c in X.columns]}
    if sp.task == "classification":
        prev["class_distribution"] = {"train": {str(k): int(v) for k, v in ytr.value_counts().items()}, "test": {str(k): int(v) for k, v in yte.value_counts().items()}}
    ctx.log(f"Split: {len(Xtr)} train / {len(Xte)} test (seed {p['seed']}, shuffle={shuffle}, stratified={strat is not None})")
    return NodeResult({"split": sp}, prev, warns)


def code_split(c, nid, p, ins):
    sv, var = ins["dataset"][0], c.out(nid, "split")
    meta = c.meta[sv]
    if not meta.get("target"):
        raise NodeError("No target defined before the split.")
    c.imports.add("from sklearn.model_selection import train_test_split")
    tcol, shuffle = p["time_column"], p["shuffle"] and not p["time_column"]
    strat = bool(p["stratify"] and meta["task"] == "classification" and shuffle)
    pre = f"df_{sv}" if not tcol else f"df_{sv}.sort_values({tcol!r}, kind='stable')"
    code = (f"X_{var} = {pre}.drop(columns=[{meta['target']!r}])\ny_{var} = {pre}[{meta['target']!r}]\n"
            f"X_train_{var}, X_test_{var}, y_train_{var}, y_test_{var} = train_test_split(\n    X_{var}, y_{var}, test_size={p['test_size']}, random_state={p['seed']}, shuffle={shuffle}"
            + (f", stratify=y_{var}" if strat else "") + f")\nsteps_{var} = []  # preprocessing steps are appended below and fitted on the training data only\n"
            f"print(len(X_train_{var}), 'train rows /', len(X_test_{var}), 'test rows')")
    c.meta[var] = {"task": meta["task"], "target": meta["target"], "seed": p["seed"], "time_ordered": bool(tcol)}
    c.cell(nid, "Train/test split", "Preprocessing is **fitted on the training split only** and then applied to the test split, preventing data leakage.", code)
    return {"split": var}


register(NodeSpec("train_test_split", "Train/Test Split", "Data Splitting", "Hold out a test set. Supports stratification and chronological splits.", DS_IN, SP_OUT,
                  [Param("test_size", "Test size", "float", 0.2, "Fraction held out for the final evaluation.", min=0.05, max=0.9, step=0.05),
                   Param("seed", "Random seed", "int", 42, "Same seed = same split.", min=0),
                   Param("shuffle", "Shuffle", "bool", True, "Randomly shuffle before splitting."),
                   Param("stratify", "Stratify by target", "bool", True, "Keeps class proportions (classification only).", visible_if={"shuffle": [True]}),
                   Param("time_column", "Time column (optional)", "column", "", "If set, rows are ordered chronologically and never shuffled.", advanced=True)],
                  run_split, code_split, icon="split",
                  doc={"what": "Splits rows into a training set (to learn from) and a test set (to evaluate honestly).",
                       "how": "All later preprocessing nodes are fitted on training data only. With a time column the split is chronological.",
                       "when_not_to_use": "Do not use random shuffling for time-series data.", "limitations": "A single split can be noisy; use cross-validation on the model node for a steadier estimate."},
                  keywords="train test validation holdout stratified time series"))


# ------------------------------------------------------------------ Preprocessing steps (attached to the split)
def _preview_step(split: Split, new: Split, ctx, warns):
    from ..workflow.executor import friendly
    try:
        pre = build_pipeline(new.steps, new.task)
        Xt = pre.fit_transform(new.X_train, new.y_train)
    except NodeError:
        raise
    except Exception as e:  # noqa: BLE001
        raise friendly(e)
    before_miss = int(split.X_train.isna().sum().sum())
    after_miss = int(Xt.isna().sum().sum())
    return {"kind": "preprocess", "columns_before": [str(c) for c in split.X_train.columns], "columns_after": [str(c) for c in Xt.columns],
            "missing_before": before_miss, "missing_after": after_miss, "train_preview": df_preview(Xt, 20), "columns": [str(c) for c in Xt.columns],
            "n_steps": len(new.steps), "fit_on": "training rows only", "n_train": len(Xt)}


def make_step_node(kind, name, desc, params, doc, icon, extra_check=None, keywords=""):
    def run(ctx, p, ins):
        sp = ins["split"][0]
        step = {"kind": kind, "params": p, "seed": sp.seed}
        new = sp.with_step(step)
        pv = _preview_step(sp, new, ctx, [])
        ctx.log(f"{name}: {pv['missing_before']} → {pv['missing_after']} missing; {len(pv['columns_before'])} → {len(pv['columns_after'])} columns (fitted on train only)")
        return NodeResult({"split": new}, pv, [])

    def code(c, nid, p, ins):
        sv, var = ins["split"][0], c.out(nid, "split")
        meta = c.meta[sv]
        expr, imps = step_code({"kind": kind, "params": p, "seed": meta["seed"]}, meta["task"])
        c.imports |= imps
        c.meta[var] = dict(meta)
        c.cell(nid, name, f"{desc}\n\nStep is added to a pipeline: fitted on training data only.",
               f"X_train_{var}, X_test_{var}, y_train_{var}, y_test_{var} = X_train_{sv}, X_test_{sv}, y_train_{sv}, y_test_{sv}\n"
               f"steps_{var} = steps_{sv} + [({f's{kind}_{nid}'!r}, {expr})]")
        return {"split": var}
    register(NodeSpec(f"pp_{kind}", name, "Feature Selection" if kind == "feature_select" else "Data Preprocessing", desc, SP_IN, SP_OUT, params, run, code, icon=icon, doc=doc, keywords=keywords))


make_step_node("imputer", "Imputer", "Fill in missing values.",
               [Param("strategy", "Strategy", "select", "median", "How to fill gaps.", options=["mean", "median", "most_frequent", "constant", "knn"]),
                Param("apply_to", "Apply to", "select", "numeric", "Numerical or categorical columns.", options=["numeric", "categorical"]),
                Param("fill_value", "Constant value", "string", "0", "Used by 'constant'.", visible_if={"strategy": ["constant"]}),
                Param("n_neighbors", "Neighbours", "int", 5, "KNN imputer only.", min=1, max=50, visible_if={"strategy": ["knn"]}),
                Param("columns", "Specific columns (optional)", "columns", [], "Empty = all columns of the chosen type.", advanced=True)],
               {"what": "Replaces missing values with the mean/median/most-frequent/constant or KNN estimate.", "how": "The fill values are learned from the TRAINING rows only and reused on the test rows.",
                "when_to_use": "Whenever features contain gaps and the model cannot handle NaN.", "limitations": "Median is robust to outliers; mean is not. KNN is slower."}, "wrench", keywords="missing nan fill mean median impute")
make_step_node("scaler", "Scaler / Transformer", "Scale or transform numerical features.",
               [Param("method", "Method", "select", "standard", "Transformation.", options=["standard", "minmax", "robust", "maxabs", "power", "quantile", "normalizer", "polynomial", "log1p"]),
                Param("degree", "Polynomial degree", "int", 2, "", min=2, max=5, visible_if={"method": ["polynomial"]}),
                Param("columns", "Specific columns (optional)", "columns", [], "Empty = all numerical columns.", advanced=True)],
               {"what": "StandardScaler, MinMax, Robust, MaxAbs, Normalizer, PowerTransformer, QuantileTransformer, PolynomialFeatures or log1p.", "how": "Statistics (mean, std, min…) are learned on training rows only.",
                "when_to_use": "Distance-, margin- and gradient-based models (KNN, SVM, linear, MLP). Tree models don't need it.", "limitations": "log1p needs values > -1. Polynomial features grow quickly."},
               "scale", keywords="standard scaler minmax robust normalize polynomial log power quantile")
make_step_node("encoder", "Categorical Encoder", "Convert categorical columns to numbers.",
               [Param("method", "Method", "select", "onehot", "One-hot creates a 0/1 column per category; ordinal maps categories to integers.", options=["onehot", "ordinal"]),
                Param("drop_first", "Drop first category", "bool", False, "Avoids perfect collinearity for linear models.", visible_if={"method": ["onehot"]}),
                Param("columns", "Specific columns (optional)", "columns", [], "Empty = all non-numerical columns.", advanced=True)],
               {"what": "One-hot or ordinal encoding of feature columns (the target is handled separately and never encoded here).", "how": "Categories are learned from training rows; unseen categories in test data are handled safely.",
                "when_to_use": "Before any model that needs numbers.", "limitations": "One-hot on high-cardinality columns explodes the column count."},
               "tags", keywords="onehot ordinal categorical encode")
make_step_node("feature_select", "Feature Selection", "Keep only the most useful features.",
               [Param("method", "Method", "select", "select_k_best", "", options=["select_k_best", "variance_threshold", "select_from_model", "rfe"]),
                Param("score", "Score function", "select", "f_test", "ANOVA F-test, mutual information or chi-square (classification).", options=["f_test", "mutual_info", "chi2"], visible_if={"method": ["select_k_best"]}),
                Param("k", "Features to keep", "int", 5, "Number of features kept.", min=1, max=1000, visible_if={"method": ["select_k_best", "rfe"]}),
                Param("threshold", "Variance threshold", "float", 0.0, "Drop features whose variance is ≤ this.", min=0, max=1000, step=0.01, visible_if={"method": ["variance_threshold"]})],
               {"what": "Selects features with univariate tests, variance, a model's importances or recursive elimination.", "how": "Selection is learned on training data only. The preview shows which columns survived.",
                "when_to_use": "Many features, noisy features, or to compare feature subsets.", "limitations": "Requires numeric features (encode/impute first). k larger than the feature count raises an error."},
               "filter", keywords="selectkbest variance rfe mutual information anova chi2")
