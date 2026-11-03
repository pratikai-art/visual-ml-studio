"""Graph model, topological ordering and ancestor queries."""
from __future__ import annotations
from collections import defaultdict, deque
from pydantic import BaseModel, Field
from typing import Any, Optional


class NodeModel(BaseModel):
    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    position: Optional[dict[str, float]] = None


class EdgeModel(BaseModel):
    id: Optional[str] = None
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None


class Workflow(BaseModel):
    name: str = "Untitled project"
    nodes: list[NodeModel] = Field(default_factory=list)
    edges: list[EdgeModel] = Field(default_factory=list)
    journal: dict[str, str] = Field(default_factory=dict)


class CycleError(Exception):
    def __init__(self, nodes):
        super().__init__("The workflow contains a cycle: " + " → ".join(nodes))
        self.nodes = nodes


def topo_order(wf: Workflow) -> list[str]:
    ids = [n.id for n in wf.nodes]
    indeg = {i: 0 for i in ids}
    out = defaultdict(list)
    for e in wf.edges:
        if e.source in indeg and e.target in indeg:
            out[e.source].append(e.target)
            indeg[e.target] += 1
    q = deque(i for i in ids if indeg[i] == 0)
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for m in out[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    if len(order) != len(ids):
        raise CycleError([i for i in ids if i not in order])
    return order


def ancestors(wf: Workflow, node_id: str) -> set[str]:
    parents = defaultdict(list)
    for e in wf.edges:
        parents[e.target].append(e.source)
    seen, stack = set(), [node_id]
    while stack:
        n = stack.pop()
        for p in parents[n]:
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen
