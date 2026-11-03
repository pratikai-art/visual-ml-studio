from __future__ import annotations
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import nbformat
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

from .. import nodes  # noqa: F401
from ..codegen.generator import to_notebook, to_script, validate_notebook
from ..config import MAX_UPLOAD_MB, PROJECT_DIR
from ..datasets import builtin, store
from ..experiments import history
from ..utils.jsonable import to_jsonable
from ..workflow import executor
from ..workflow.graph import Workflow, ancestors
from ..workflow.spec import CATEGORY_ORDER, REGISTRY, NodeError
from ..workflow.validator import check_connection, validate

router = APIRouter(prefix="/api")
POOL = ThreadPoolExecutor(max_workers=2)
CANCELS: dict[str, threading.Event] = {}


@router.get("/health")
def health():
    import sklearn, pandas
    return {"status": "ok", "sklearn": sklearn.__version__, "pandas": pandas.__version__, "max_upload_mb": MAX_UPLOAD_MB}


@router.get("/nodes")
def list_nodes():
    return {"categories": CATEGORY_ORDER, "nodes": [s.to_dict() for s in REGISTRY.values()]}


@router.get("/builtin-datasets")
def builtin_datasets():
    return builtin.catalog()


# ---------------------------------------------------------------- datasets
@router.post("/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    content = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    try:
        meta = store.save_upload(file.filename or "dataset.csv", content)
        df = store.load_df(meta["id"])
    except store.DatasetError as e:
        raise HTTPException(400, str(e))
    return {"meta": meta, "profile": store.profile_df(df)}


@router.get("/datasets")
def datasets():
    return store.list_datasets()


@router.get("/datasets/{dataset_id}")
def dataset_profile(dataset_id: str):
    try:
        df = store.load_df(dataset_id)
        return {"meta": store.get_meta(dataset_id), "profile": store.profile_df(df)}
    except store.DatasetError as e:
        raise HTTPException(404, str(e))


# ---------------------------------------------------------------- workflow
class ConnReq(BaseModel):
    source_type: str
    source_port: str
    target_type: str
    target_port: str


@router.post("/workflow/check-connection")
def conn(req: ConnReq):
    why = check_connection(req.source_type, req.source_port, req.target_type, req.target_port)
    return {"valid": why is None, "reason": why}


@router.post("/workflow/validate")
def validate_wf(wf: Workflow):
    issues = validate(wf)
    return {"valid": not any(i["level"] == "error" for i in issues), "issues": issues}


class RunReq(BaseModel):
    workflow: Workflow
    target_node: str | None = None


@router.post("/workflow/run")
def run_wf(req: RunReq):
    job = executor.new_job(req.workflow, req.target_node)
    ev = threading.Event()
    CANCELS[job["id"]] = ev
    POOL.submit(executor.execute, req.workflow, job, ev, req.target_node)
    return {"job_id": job["id"]}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = executor.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    with executor._LOCK:
        return JSONResponse(to_jsonable(job))


@router.post("/jobs/{job_id}/cancel")
def cancel(job_id: str):
    ev = CANCELS.get(job_id)
    if not ev:
        raise HTTPException(404, "Job not found.")
    ev.set()
    return {"cancelling": True, "note": "Stops at the next node boundary; a model that is mid-fit finishes first."}


class ColReq(BaseModel):
    workflow: Workflow
    node_id: str


@router.post("/workflow/columns")
def columns(req: ColReq):
    """Columns that flow INTO a node (for the searchable column selectors)."""
    wf = req.workflow
    srcs = [e.source for e in wf.edges if e.target == req.node_id]
    if not srcs:
        return {"columns": [], "message": "Connect an upstream dataset node to see its columns."}
    src = srcs[0]
    job = executor.new_job(wf, src)
    executor.execute(wf, job, threading.Event(), src)
    st = job["nodes"].get(src, {})
    pv = st.get("preview") or {}
    if st.get("status") not in ("succeeded", "warning"):
        msg = (st.get("error") or {}).get("message") or "Upstream nodes have errors; fix them to see columns."
        return {"columns": [], "message": msg}
    tbl = pv.get("table") or pv.get("train_preview") or {}
    names, dtypes = pv.get("columns", []), tbl.get("dtypes", [])
    out = [{"name": n, "dtype": dtypes[i] if i < len(dtypes) and tbl.get("columns", [None])[i] == n else ""} for i, n in enumerate(names)]
    return {"columns": out, "suggested_target": pv.get("suggested_target")}


# ---------------------------------------------------------------- code
@router.post("/workflow/code")
def code(wf: Workflow, node_id: str | None = None):
    try:
        return {"code": to_script(wf, node_id)}
    except NodeError as e:
        raise HTTPException(400, e.message)


@router.post("/workflow/script")
def script(wf: Workflow):
    try:
        src = to_script(wf)
    except NodeError as e:
        raise HTTPException(400, e.message)
    return PlainTextResponse(src, headers={"Content-Disposition": 'attachment; filename="experiment.py"'})


@router.post("/workflow/notebook")
def notebook(wf: Workflow):
    try:
        nb = to_notebook(wf)
    except NodeError as e:
        raise HTTPException(400, e.message)
    report = validate_notebook(nb)
    if not report["valid"]:
        raise HTTPException(500, "Generated notebook failed validation: " + "; ".join(report["problems"]))
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", wf.name)[:50] or "experiment"
    return Response(nbformat.writes(nb), media_type="application/x-ipynb+json",
                    headers={"Content-Disposition": f'attachment; filename="{name}.ipynb"', "X-Notebook-Valid": "true", "X-Notebook-Cells": str(report["cells"])})


# ---------------------------------------------------------------- projects
FORMAT = "visual-ml-studio-project"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-")[:60] or "project"


@router.post("/projects")
def save_project(project: dict):
    if project.get("format") != FORMAT:
        raise HTTPException(400, "Not a Visual ML Studio project.")
    slug = _slug(project.get("name", "project"))
    (PROJECT_DIR / f"{slug}.json").write_text(json.dumps(project))
    return {"saved": slug}


@router.get("/projects")
def list_projects():
    out = []
    for p in sorted(PROJECT_DIR.glob("*.json"), key=lambda x: -x.stat().st_mtime):
        try:
            d = json.loads(p.read_text())
            out.append({"slug": p.stem, "name": d.get("name"), "modified": p.stat().st_mtime, "nodes": len(d.get("nodes", []))})
        except Exception:  # noqa: BLE001
            continue
    return out


@router.get("/projects/{slug}")
def load_project(slug: str):
    p = PROJECT_DIR / f"{_slug(slug)}.json"
    if not p.exists():
        raise HTTPException(404, "Project not found.")
    return json.loads(p.read_text())


@router.post("/projects/check")
def check_project(project: dict):
    """Report dataset files that a project refers to but the server does not have."""
    missing, ok = [], []
    for n in project.get("nodes", []):
        prm = n.get("data", {}).get("params", n.get("params", {})) if isinstance(n, dict) else {}
        if n.get("type") == "data_input" or n.get("data", {}).get("type") == "data_input":
            if prm.get("source") == "upload" and prm.get("dataset_id"):
                try:
                    store.get_meta(prm["dataset_id"])
                    ok.append(prm["dataset_id"])
                except store.DatasetError:
                    ref = next((d for d in project.get("datasets", []) if d.get("id") == prm["dataset_id"]), {})
                    missing.append({"node_id": n.get("id"), "dataset_id": prm["dataset_id"], "filename": ref.get("filename", "unknown")})
    return {"missing": missing, "ok": ok}


# ---------------------------------------------------------------- history
@router.get("/history")
def get_history():
    return history.list_history()


@router.delete("/history")
def clear_history():
    history.clear_history()
    return {"cleared": True}
