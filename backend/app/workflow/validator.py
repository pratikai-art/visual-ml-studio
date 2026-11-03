"""Server-side workflow validation (the frontend is never the only enforcement layer)."""
from __future__ import annotations
from collections import defaultdict
from .graph import CycleError, Workflow, topo_order
from .spec import REGISTRY, validate_params
from .. import nodes  # noqa: F401  (registers nodes)


def check_connection(src_type: str, src_port: str, dst_type: str, dst_port: str) -> str | None:
    """Return None if compatible, else a human-readable reason."""
    s, d = REGISTRY.get(src_type), REGISTRY.get(dst_type)
    if not s or not d:
        return "Unknown node type."
    so = next((p for p in s.outputs if p.name == src_port), None)
    di = next((p for p in d.inputs if p.name == dst_port), None)
    if so is None:
        return f"'{s.name}' has no output port '{src_port}'."
    if di is None:
        return f"'{d.name}' has no input port '{dst_port}'."
    if so.type != di.type:
        hint = {("dataset", "trained"): "Add a Train/Test Split and a model first.", ("dataset", "split"): "Insert a Train/Test Split node.",
                ("split", "dataset"): "Splits cannot go back to dataset nodes.", ("trained", "metrics"): "Add a Model Evaluation node.",
                ("split", "metrics"): "Train a model before evaluating."}.get((so.type, di.type), "")
        return f"A {so.type} output cannot connect to a {di.type} input ('{d.name}' needs {di.type}). {hint}".strip()
    return None


def validate(wf: Workflow, only: set[str] | None = None) -> list[dict]:
    issues: list[dict] = []
    ids = {n.id: n for n in wf.nodes}
    if len(ids) != len(wf.nodes):
        issues.append({"level": "error", "message": "Duplicate node ids."})
    for n in wf.nodes:
        if only is not None and n.id not in only:
            continue
        spec = REGISTRY.get(n.type)
        if not spec:
            issues.append({"level": "error", "node_id": n.id, "message": f"Unknown node type '{n.type}'."})
            continue
        _, errs = validate_params(spec, n.params)
        for e in errs:
            issues.append({"level": "error", "node_id": n.id, "message": e})
    incoming = defaultdict(lambda: defaultdict(list))
    for e in wf.edges:
        if e.source not in ids or e.target not in ids:
            issues.append({"level": "error", "edge_id": e.id, "message": "Connection refers to a node that does not exist."})
            continue
        why = check_connection(ids[e.source].type, e.sourceHandle or "", ids[e.target].type, e.targetHandle or "")
        if why:
            issues.append({"level": "error", "node_id": e.target, "edge_id": e.id, "message": why})
        else:
            incoming[e.target][e.targetHandle].append(e.source)
    for n in wf.nodes:
        if only is not None and n.id not in only:
            continue
        spec = REGISTRY.get(n.type)
        if not spec:
            continue
        for port in spec.inputs:
            srcs = incoming[n.id].get(port.name, [])
            if not srcs and port.required:
                issues.append({"level": "error", "node_id": n.id, "message": f"Missing required input '{port.label or port.name}' ({port.type}). Connect an upstream node."})
            if len(srcs) > 1 and not port.multi:
                issues.append({"level": "error", "node_id": n.id, "message": f"Input '{port.label or port.name}' accepts only one connection."})
    try:
        topo_order(wf)
    except CycleError as e:
        issues.append({"level": "error", "message": str(e), "node_id": e.nodes[0] if e.nodes else None})
    if not any(n.type == "data_input" for n in wf.nodes) and wf.nodes:
        issues.append({"level": "warning", "message": "The workflow has no Dataset Input node."})
    return issues
