"""Helpers that turn numpy / pandas / plotly objects into strict JSON (no NaN/Inf)."""
from __future__ import annotations
import json
import math
from typing import Any
import numpy as np
import pandas as pd


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.tolist())
    return str(obj)


def df_preview(df: pd.DataFrame, n: int = 100) -> dict:
    head = df.head(n)
    rows = [[to_jsonable(v) for v in r] for r in head.itertuples(index=False, name=None)]
    return {
        "columns": [str(c) for c in df.columns],
        "dtypes": [str(t) for t in df.dtypes],
        "rows": rows,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "shown": int(len(head)),
    }


def _decode_bdata(o):
    """Plotly >= 6 emits {"dtype","bdata"[,"shape"]} typed arrays; decode to plain lists for any Plotly.js version."""
    import base64
    if isinstance(o, dict):
        if "bdata" in o and "dtype" in o:
            arr = np.frombuffer(base64.b64decode(o["bdata"]), dtype=np.dtype(o["dtype"]))
            if "shape" in o:
                shp = o["shape"]
                arr = arr.reshape([int(x) for x in (shp.split(",") if isinstance(shp, str) else shp)])
            return arr.tolist()
        return {k: _decode_bdata(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_decode_bdata(v) for v in o]
    return o


def fig_json(fig) -> dict:
    return to_jsonable(_decode_bdata(to_jsonable(fig.to_plotly_json())))
