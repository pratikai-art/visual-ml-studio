"""Graph executor: validates, plans, runs nodes in dependency order with caching."""
from __future__ import annotations
import threading
import time
import traceback
import uuid
from collections import defaultdict

from ..config import JOB_TIMEOUT_S
from .cache import ExecutionCache
from .graph import Workflow, ancestors, topo_order
from .spec import REGISTRY, NodeError, validate_params
from .validator import validate
from ..utils.jsonable import to_jsonable

CACHE = ExecutionCache()
JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


class Cancelled(Exception):
    pass


class Ctx:
    def __init__(self, job, node_id, cancel):
        self.job, self.node_id, self.cancel = job, node_id, cancel

    def log(self, msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with _LOCK:
            self.job["nodes"][self.node_id]["logs"].append(line)
            self.job["logs"].append({"node_id": self.node_id, "text": line})

    def check_cancel(self):
        if self.cancel.is_set():
            raise Cancelled()


def friendly(e: Exception) -> NodeError:
    """Translate common library errors into beginner-friendly messages."""
    s = str(e)
    tech = "".join(traceback.format_exception_only(type(e), e)).strip()
    if "could not convert string to float" in s or "non-numeric data" in s or "Cannot use mean strategy" in s or "Cannot use median strategy" in s:
        return NodeError("The model received categorical data that has not been encoded. Connect an encoder or configure a preprocessing pipeline.", tech)
    if "contains NaN" in s or "Input X contains" in s or "Input contains NaN" in s:
        return NodeError("The data still contains missing values. Add an Imputer node before this step (or drop the incomplete rows).", tech)
    if "n_features" in s and "k" in s and "SelectKBest" in s or "k should be" in s:
        return NodeError("'Features to keep' is larger than the number of available features. Lower k.", tech)
    if "Negative values in data passed to" in s or "Input X must be non-negative" in s:
        return NodeError("This step requires non-negative values (e.g. chi-square or Multinomial NB). Use MinMax scaling or choose another score function.", tech)
    if "invalid value encountered in log1p" in s or "log1p" in s:
        return NodeError("log1p needs every value to be greater than -1. Remove or shift negative values first.", tech)
    return NodeError(f"{type(e).__name__}: {s[:300]}", "".join(traceback.format_exception(type(e), e, e.__traceback__))[-3000:])


def new_job(wf: Workflow, target: str | None) -> dict:
    job = {"id": uuid.uuid4().hex[:12], "status": "Idle", "phase": "Idle", "workflow": wf.name, "target": target, "started": time.time(), "finished": None,
           "progress": 0.0, "logs": [], "issues": [], "nodes": {n.id: {"status": "not_run", "duration": None, "logs": [], "warnings": [], "error": None, "preview": None, "cached": False}
                                                                 for n in wf.nodes}}
    with _LOCK:
        JOBS[job["id"]] = job
        for k in list(JOBS)[:-30]:
            JOBS.pop(k, None)
    return job


def _phase(node_type: str) -> str:
    cat = REGISTRY[node_type].category
    return {"Machine Learning": "Training", "Clustering": "Training", "Evaluation": "Evaluating", "Visualization": "Evaluating"}.get(cat, "Preparing Data")


def execute(wf: Workflow, job: dict, cancel: threading.Event, target: str | None = None) -> None:
    def setn(nid, **kw):
        with _LOCK:
            job["nodes"][nid].update(kw)

    def setj(**kw):
        with _LOCK:
            job.update(kw)

    try:
        setj(status="Validating", phase="Validating")
        plan_ids = None
        if target:
            if target not in {n.id for n in wf.nodes}:
                raise NodeError("The selected node does not exist.")
            plan_ids = ancestors(wf, target) | {target}
        issues = validate(wf, plan_ids)
        setj(issues=issues)
        errs = [i for i in issues if i["level"] == "error"]
        if errs:
            for i in errs:
                if i.get("node_id") in job["nodes"]:
                    setn(i["node_id"], status="failed", error={"message": i["message"], "technical": None})
            setj(status="Failed", phase="Failed", finished=time.time())
            return
        order = [i for i in topo_order(wf) if plan_ids is None or i in plan_ids]
        nodes = {n.id: n for n in wf.nodes}
        by_target = defaultdict(list)
        for e in wf.edges:
            by_target[e.target].append(e)
        outputs: dict = {}
        keys: dict = {}
        ok: dict = {}
        started = time.time()
        for idx, nid in enumerate(order):
            if cancel.is_set():
                raise Cancelled()
            if time.time() - started > JOB_TIMEOUT_S:
                raise NodeError(f"The workflow exceeded the {JOB_TIMEOUT_S}s time limit.")
            n = nodes[nid]
            spec = REGISTRY[n.type]
            params, _ = validate_params(spec, n.params)
            edges = by_target[nid]
            if any(not ok.get(e.source) for e in edges):
                setn(nid, status="skipped", error={"message": "Skipped because an upstream node did not succeed.", "technical": None})
                ok[nid] = False
                continue
            ins: dict = defaultdict(list)
            for e in edges:
                ins[e.targetHandle].append(outputs[(e.source, e.sourceHandle)])
            ukeys = sorted(f"{e.targetHandle}:{keys[e.source]}" for e in edges)
            key = CACHE.key(n.type, params, ukeys)
            keys[nid] = key
            setj(phase=_phase(n.type), status=_phase(n.type), progress=idx / max(len(order), 1))
            cached = CACHE.get(key)
            if cached is not None:
                res, preview, warns = cached
                for port, obj in res.items():
                    outputs[(nid, port)] = obj
                setn(nid, status="warning" if warns else "succeeded", duration=0.0, preview=preview, warnings=warns, cached=True)
                Ctx(job, nid, cancel).log(f"{spec.name}: reused cached result (inputs and settings unchanged)")
                ok[nid] = True
                continue
            setn(nid, status="running", cached=False, error=None)
            ctx = Ctx(job, nid, cancel)
            ctx.log(f"Running {spec.name}…")
            t0 = time.perf_counter()
            try:
                r = spec.run(ctx, params, dict(ins))
                dur = time.perf_counter() - t0
                preview = to_jsonable(r.preview)
                for port, obj in r.outputs.items():
                    outputs[(nid, port)] = obj
                CACHE.put(key, (r.outputs, preview, r.warnings))
                setn(nid, status="warning" if r.warnings else "succeeded", duration=dur, preview=preview, warnings=r.warnings)
                ctx.log(f"{spec.name} finished in {dur:.2f}s")
                ok[nid] = True
            except Cancelled:
                raise
            except NodeError as e:
                setn(nid, status="failed", duration=time.perf_counter() - t0, error={"message": e.message, "technical": e.technical})
                ctx.log(f"ERROR in {spec.name}: {e.message}")
                ok[nid] = False
            except Exception as e:  # noqa: BLE001
                fe = friendly(e)
                setn(nid, status="failed", duration=time.perf_counter() - t0, error={"message": fe.message, "technical": fe.technical})
                ctx.log(f"ERROR in {spec.name}: {fe.message}")
                ok[nid] = False
        failed = [i for i, v in ok.items() if not v and job["nodes"][i]["status"] == "failed"]
        setj(status="Failed" if failed else "Completed", phase="Failed" if failed else "Completed", progress=1.0, finished=time.time())
        if not failed:
            from ..experiments.history import record_run
            record_run(job, wf)
    except Cancelled:
        for nid, st in job["nodes"].items():
            if st["status"] == "running":
                setn(nid, status="skipped", error={"message": "Stopped by user.", "technical": None})
        setj(status="Failed", phase="Failed", finished=time.time(), cancelled=True)
        job["logs"].append({"node_id": None, "text": "Execution was stopped. Results from nodes that already finished are preserved."})
    except Exception as e:  # noqa: BLE001
        setj(status="Failed", phase="Failed", finished=time.time())
        job["logs"].append({"node_id": None, "text": f"Workflow error: {e}"})
