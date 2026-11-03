import threading
from app.workflow.executor import execute, new_job
from app.workflow.graph import Workflow


def N(i, t, **p): return {"id": i, "type": t, "params": p}
def E(s, sh, t, th): return {"source": s, "sourceHandle": sh, "target": t, "targetHandle": th}


def run(wf_dict, target=None):
    wf = Workflow(**wf_dict)
    job = new_job(wf, target)
    execute(wf, job, threading.Event(), target)
    return job
