"""Content-addressed dataset storage, safe loading and profiling."""
from __future__ import annotations
import hashlib
import json
import re
import time
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ALLOWED_EXT, MAX_UPLOAD_MB, UPLOAD_DIR
from ..utils.jsonable import df_preview, to_jsonable
from ..workflow.spec import NodeError


class DatasetError(Exception):
    pass


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)[:120] or "dataset"


def _csv_separator(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:32768]
    lines = [line for line in sample.splitlines() if line.strip()][:20]
    counts = {delimiter: sum(line.count(delimiter) for line in lines) for delimiter in ",;\t|"}
    best, count = max(counts.items(), key=lambda item: item[1])
    if count:
        return best
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_file(path: Path, ext: str, delimiter: str | None = None) -> pd.DataFrame:
    if ext == ".csv":
        return pd.read_csv(path, sep=delimiter or _csv_separator(path))
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext == ".xlsx":
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)
    if ext == ".jsonl":
        return pd.read_json(path, lines=True)
    raise DatasetError(f"Unsupported file type {ext}")


def save_upload(filename: str, content: bytes) -> dict:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise DatasetError(f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXT))}.")
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise DatasetError(f"File is larger than the {MAX_UPLOAD_MB} MB limit.")
    if not content:
        raise DatasetError("The file is empty.")
    digest = hashlib.sha256(content).hexdigest()[:16]
    path = UPLOAD_DIR / f"{digest}{ext}"
    path.write_bytes(content)
    delimiter = _csv_separator(path) if ext == ".csv" else None
    try:
        df = read_file(path, ext, delimiter)
    except Exception as e:  # noqa: BLE001
        path.unlink(missing_ok=True)
        raise DatasetError(f"Could not parse the file as {ext}: {e}") from e
    meta = {"id": digest, "filename": _safe_name(filename), "ext": ext, "delimiter": delimiter, "size": len(content),
            "rows": int(df.shape[0]), "cols": int(df.shape[1]), "uploaded_at": time.time()}
    (UPLOAD_DIR / f"{digest}.json").write_text(json.dumps(meta))
    return meta


def get_meta(dataset_id: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{16}", dataset_id or ""):
        raise DatasetError("Invalid dataset id.")
    p = UPLOAD_DIR / f"{dataset_id}.json"
    if not p.exists():
        raise DatasetError(f"Dataset '{dataset_id}' was not found. Upload it again or reconnect it.")
    return json.loads(p.read_text())


def list_datasets() -> list:
    return sorted((json.loads(p.read_text()) for p in UPLOAD_DIR.glob("*.json")),
                  key=lambda m: -m["uploaded_at"])


def load_df(dataset_id: str) -> pd.DataFrame:
    meta = get_meta(dataset_id)
    path = UPLOAD_DIR / f"{dataset_id}{meta['ext']}"
    delimiter = meta.get("delimiter")
    df = read_file(path, meta["ext"], delimiter)
    # Older uploads did not store their delimiter. If they were parsed as one
    # column, retry delimiter detection so existing projects load correctly.
    if meta["ext"] == ".csv" and len(df.columns) == 1:
        detected = _csv_separator(path)
        if detected != (delimiter or ","):
            df = read_file(path, meta["ext"], detected)
            meta["delimiter"] = detected
            meta["rows"] = int(df.shape[0])
            meta["cols"] = int(df.shape[1])
            (UPLOAD_DIR / f"{dataset_id}.json").write_text(json.dumps(meta))
    return df


def infer_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    n = len(s.dropna())
    nun = s.nunique(dropna=True)
    if pd.api.types.is_numeric_dtype(s):
        if n and nun <= min(10, max(2, int(0.05 * n))) and (s.dropna() % 1 == 0).all():
            return "categorical (numeric codes)"
        return "numerical"
    if n and nun / n > 0.95 and n > 20:
        return "identifier / free text"
    if n and nun <= max(20, int(0.5 * n)):
        return "categorical"
    return "text"


def profile_df(df: pd.DataFrame, preview_rows: int = 50) -> dict:
    n = len(df)
    cols, warnings = [], []
    for c in df.columns:
        s = df[c]
        kind = infer_kind(s)
        miss = int(s.isna().sum())
        cols.append({"name": str(c), "dtype": str(s.dtype), "kind": kind, "missing": miss,
                     "missing_pct": round(100 * miss / n, 2) if n else 0.0,
                     "unique": int(s.nunique(dropna=True))})
        if n and miss / n > 0.4:
            warnings.append(f"Column '{c}' is {100*miss/n:.0f}% missing.")
        if s.nunique(dropna=True) <= 1:
            warnings.append(f"Column '{c}' is constant and carries no information.")
        if kind == "identifier / free text":
            warnings.append(f"Column '{c}' looks like an identifier; exclude it from the features.")
        if s.dtype == object:
            types = {type(v).__name__ for v in s.dropna().head(500)}
            if len(types) > 1:
                warnings.append(f"Column '{c}' mixes data types ({', '.join(sorted(types))}).")
    num = df.select_dtypes(include="number")
    stats = {}
    if not num.empty:
        d = num.describe().T
        stats = {str(i): {k: to_jsonable(v) for k, v in row.items()} for i, row in d.iterrows()}
    dups = int(df.duplicated().sum())
    if dups:
        warnings.append(f"{dups} duplicate rows found.")
    mem_mb = float(df.memory_usage(deep=True).sum() / 1e6)
    if mem_mb > 200:
        warnings.append(f"Dataset uses ~{mem_mb:.0f} MB in memory; consider sampling.")
    cands = []
    for c in cols:
        low = c["name"].lower()
        if low in ("target", "label", "class", "y", "outcome") or any(k in low for k in ("score", "price", "survived", "species")):
            cands.append(c["name"])
    if cols and cols[-1]["name"] not in cands:
        cands.append(cols[-1]["name"])
    return {
        "rows": n, "cols": int(df.shape[1]), "columns": cols, "stats": stats,
        "duplicates": dups, "memory_mb": round(mem_mb, 3),
        "numerical": [c["name"] for c in cols if c["kind"] == "numerical"],
        "categorical": [c["name"] for c in cols if c["kind"].startswith("categorical")],
        "possible_targets": cands, "warnings": warnings, "preview": df_preview(df, preview_rows),
    }
