from __future__ import annotations
import json
import threading
import time
from ..config import HISTORY_FILE

_L = threading.Lock()


def _load() -> list:
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:  # noqa: BLE001
        return []


def record_run(job: dict, wf) -> None:
    entries = []
    for nid, st in job["nodes"].items():
        pv = st.get("preview") or {}
        if pv.get("kind") == "metrics":
            entries.append({k: pv.get(k) for k in ("model", "node", "task", "metrics", "train_metrics", "cv", "train_time", "infer_time", "params", "fingerprint", "seed", "n_train", "n_test")})
        elif pv.get("kind") == "clustering":
            entries.append({"model": "clustering", "node": nid, "task": "clustering", "metrics": {k: v for k, v in pv["metrics"].items() if isinstance(v, float)},
                            "params": {}, "train_time": st.get("duration"), "fingerprint": ""})
    if not entries:
        return
    rec = {"run_id": job["id"], "time": time.time(), "workflow": wf.name, "duration": (job["finished"] or time.time()) - job["started"], "models": entries}
    with _L:
        h = _load()
        h.insert(0, rec)
        HISTORY_FILE.write_text(json.dumps(h[:200]))


def list_history() -> list:
    with _L:
        return _load()


def clear_history() -> None:
    with _L:
        HISTORY_FILE.write_text("[]")
