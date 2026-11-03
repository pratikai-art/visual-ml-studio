import { create } from "zustand";
import { applyNodeChanges, applyEdgeChanges, type Node, type Edge, type NodeChange, type EdgeChange, type Connection } from "@xyflow/react";
import { api } from "./api";
import { TEMPLATES } from "./templates";
import type { Job, NodeSpec, Issue } from "./types";

export type RFNode = Node<{ type: string; params: Record<string, any> }>;
type Snap = { nodes: RFNode[]; edges: Edge[] };
export type Toast = { id: number; text: string; kind: "info" | "error" | "ok" };
let uid = 1, tid = 1, poller: any = null;
const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x));
const strip = (n: RFNode) => ({ ...n, selected: false, dragging: false });

interface S {
  specs: Record<string, NodeSpec>; categories: string[]; ready: boolean;
  nodes: RFNode[]; edges: Edge[]; projectName: string; mode: "beginner" | "advanced";
  selectedId: string | null; job: Job | null; issues: Issue[]; toasts: Toast[]; journal: Record<string, string>;
  past: Snap[]; future: Snap[]; clipboard: RFNode[]; bottomTab: string; bottomH: number; datasets: any[]; dirty: boolean;
  init: () => Promise<void>; toast: (text: string, kind?: Toast["kind"]) => void; dismiss: (id: number) => void;
  addNode: (type: string, pos: { x: number; y: number }) => void; setParams: (id: string, patch: Record<string, any>, replace?: boolean) => void;
  addDuplicateRemovalStep: () => void;
  onNodesChange: (c: NodeChange[]) => void; onEdgesChange: (c: EdgeChange[]) => void; onConnect: (c: Connection) => void;
  onReconnect: (oldEdge: Edge, c: Connection) => void; disconnect: (id: string) => void; isValid: (c: Connection | Edge, ignoreEdgeId?: string) => boolean;
  select: (id: string | null) => void; commit: () => void; undo: () => void; redo: () => void;
  deleteSelected: () => void; duplicateSelected: () => void; copy: () => void; paste: () => void; autoLayout: () => void;
  loadTemplate: (i: number) => void; newProject: () => void; setName: (n: string) => void; setMode: (m: "beginner" | "advanced") => void;
  workflow: () => any; run: (target?: string) => Promise<void>; cancel: () => Promise<void>; validate: () => Promise<boolean>;
  exportProject: () => any; importProject: (p: any) => Promise<void>; setTab: (t: string) => void; setBottomH: (h: number) => void; setJournal: (k: string, v: string) => void; refreshDatasets: () => Promise<void>;
}

export const useStore = create<S>((set, get) => ({
  specs: {}, categories: [], ready: false, nodes: [], edges: [], projectName: "Untitled project", mode: "beginner", selectedId: null, job: null, issues: [], toasts: [], journal: {},
  past: [], future: [], clipboard: [], bottomTab: "logs", bottomH: 300, datasets: [], dirty: false,
  async init() {
    const r = await api.nodes();
    const specs: Record<string, NodeSpec> = {}; r.nodes.forEach((n: NodeSpec) => (specs[n.type] = n));
    set({ specs, categories: r.categories, ready: true });
    get().loadTemplate(0); get().refreshDatasets();
  },
  async refreshDatasets() { try { set({ datasets: await api.datasets() }); } catch {} },
  toast(text, kind = "info") { const id = tid++; set((s) => ({ toasts: [...s.toasts, { id, text, kind }] })); setTimeout(() => get().dismiss(id), kind === "error" ? 8000 : 3500); },
  dismiss(id) { set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })); },
  commit() { const { nodes, edges, past } = get(); set({ past: [...past.slice(-60), clone({ nodes: nodes.map(strip), edges })], future: [], dirty: true }); },
  undo() { const { past, nodes, edges, future } = get(); if (!past.length) return; const p = past[past.length - 1]; set({ past: past.slice(0, -1), future: [clone({ nodes: nodes.map(strip), edges }), ...future], nodes: p.nodes, edges: p.edges }); },
  redo() { const { future, nodes, edges, past } = get(); if (!future.length) return; const f = future[0]; set({ future: future.slice(1), past: [...past, clone({ nodes: nodes.map(strip), edges })], nodes: f.nodes, edges: f.edges }); },
  addNode(type, pos) {
    const spec = get().specs[type]; if (!spec) return; get().commit();
    const id = `n${uid++}_${Math.random().toString(36).slice(2, 5)}`;
    set((s) => ({ nodes: [...s.nodes, { id, type: "card", position: pos, data: { type, params: spec.params.reduce((a, p) => ({ ...a, [p.name]: p.default }), {}) } }], selectedId: id }));
  },
  addDuplicateRemovalStep() {
    const { nodes, edges, specs } = get();
    const input = nodes.find((n) => n.data.type === "data_input");
    const spec = specs.drop_duplicates;
    if (!input || !spec) {
      get().toast("Add a Dataset Input node before removing duplicates.", "error");
      return;
    }
    const output = specs[input.data.type]?.outputs[0];
    const targetInput = spec.inputs.find((p) => p.type === output?.type);
    if (!output || !targetInput) {
      get().toast("Could not connect the duplicate-removal step.", "error");
      return;
    }
    get().commit();
    const existing = nodes.find((n) => n.data.type === "drop_duplicates");
    const id = existing?.id ?? `n${uid++}_${Math.random().toString(36).slice(2, 5)}`;
    const node: RFNode = existing ?? {
      id,
      type: "card",
      position: { x: input.position.x + 290, y: input.position.y },
      data: { type: "drop_duplicates", params: { mode: "remove", subset: [], keep: "first" } },
    };
    const inputEdge = edges.find((e) => e.source === input.id && e.target === id);
    const outgoing = edges.filter((e) => e.source === input.id && e.id !== inputEdge?.id);
    const rewired = outgoing.map((e) => ({ ...e, source: id }));
    const duplicateEdge = inputEdge ?? { id: `e${uid++}`, source: input.id, target: id, sourceHandle: output.name, targetHandle: targetInput.name, reconnectable: true };
    set({
      nodes: existing ? nodes.map((n) => n.id === id ? { ...n, data: { ...n.data, params: { ...n.data.params, mode: "remove" } } } : n) : [...nodes, node],
      edges: [
        ...edges.filter((e) => e.source !== input.id && e.target !== id),
        duplicateEdge,
        ...rewired,
      ],
      selectedId: id,
    });
    get().toast(existing ? "Duplicate removal connected before all downstream nodes." : "Duplicate removal added to the workflow.", "ok");
  },
  setParams(id, patch, replace) { get().commit(); set((s) => ({ nodes: s.nodes.map((n) => (n.id === id ? { ...n, data: { ...n.data, params: replace ? patch : { ...n.data.params, ...patch } } } : n)) })); },
  onNodesChange(changes) {
    const finished = changes.some((c: any) => c.type === "position" && c.dragging === false);
    if (finished) get().commit();
    set((s) => ({ nodes: applyNodeChanges(changes, s.nodes) as RFNode[] }));
  },
  onEdgesChange(changes) { if (changes.some((c) => c.type === "remove")) get().commit(); set((s) => ({ edges: applyEdgeChanges(changes, s.edges) })); },
  isValid(c, ignoreEdgeId) {
    const { nodes, specs, edges } = get();
    const s = nodes.find((n) => n.id === c.source), t = nodes.find((n) => n.id === c.target); if (!s || !t || c.source === c.target) return false;
    const so = specs[s.data.type]?.outputs.find((p) => p.name === c.sourceHandle), ti = specs[t.data.type]?.inputs.find((p) => p.name === c.targetHandle);
    if (!so || !ti || so.type !== ti.type) return false;
    if (!ti.multi && edges.some((e) => e.id !== ignoreEdgeId && e.target === c.target && e.targetHandle === c.targetHandle)) return false;
    // cycle check
    const seen = new Set<string>(); const stack = [c.target as string];
    while (stack.length) { const x = stack.pop()!; if (x === c.source) return false; if (seen.has(x)) continue; seen.add(x); edges.filter((e) => e.source === x).forEach((e) => stack.push(e.target)); }
    return true;
  },
  onConnect(c) {
    const { nodes, specs, edges } = get();
    const s = nodes.find((n) => n.id === c.source)!, t = nodes.find((n) => n.id === c.target)!;
    const so = specs[s.data.type].outputs.find((p) => p.name === c.sourceHandle), ti = specs[t.data.type].inputs.find((p) => p.name === c.targetHandle);
    let why = "";
    if (!so || !ti) why = "Unknown port.";
    else if (so.type !== ti.type) why = `A ${so.type} output cannot connect to a ${ti.type} input. '${specs[t.data.type].name}' needs ${ti.type}.`;
    else if (!ti.multi && edges.some((e) => e.target === c.target && e.targetHandle === c.targetHandle)) why = `'${ti.label || ti.name}' accepts only one connection. Remove the existing one first.`;
    else if (!get().isValid(c)) why = "This connection would create a cycle.";
    if (why) { get().toast(why, "error"); return; }
    get().commit();
    set((s2) => ({ edges: [...s2.edges, { id: `e${uid++}`, source: c.source!, target: c.target!, sourceHandle: c.sourceHandle, targetHandle: c.targetHandle, animated: false, reconnectable: true }] }));
  },
  onReconnect(oldEdge, c) {
    if (!get().isValid(c, oldEdge.id)) {
      get().toast("That connection is not compatible or would create a cycle.", "error");
      return;
    }
    get().commit();
    set((s) => ({ edges: s.edges.map((e) => e.id === oldEdge.id ? { ...e, source: c.source!, target: c.target!, sourceHandle: c.sourceHandle, targetHandle: c.targetHandle } : e) }));
    get().toast("Connection rewired.", "ok");
  },
  disconnect(id) {
    if (!get().edges.some((e) => e.id === id)) return;
    get().commit();
    set((s) => ({ edges: s.edges.filter((e) => e.id !== id) }));
    get().toast("Connection removed.", "info");
  },
  select(id) { set({ selectedId: id }); },
  deleteSelected() { get().commit(); set((s) => { const ids = new Set(s.nodes.filter((n) => n.selected).map((n) => n.id)); return { nodes: s.nodes.filter((n) => !ids.has(n.id)), edges: s.edges.filter((e) => !ids.has(e.source) && !ids.has(e.target) && !e.selected), selectedId: null }; }); },
  copy() { set((s) => ({ clipboard: clone(s.nodes.filter((n) => n.selected).map(strip)) })); const n = get().clipboard.length; if (n) get().toast(`Copied ${n} node${n > 1 ? "s" : ""}`); },
  paste() {
    const { clipboard } = get(); if (!clipboard.length) return; get().commit();
    const map: Record<string, string> = {}; const fresh = clipboard.map((n) => { const id = `n${uid++}_${Math.random().toString(36).slice(2, 5)}`; map[n.id] = id; return { ...clone(n), id, selected: true, position: { x: n.position.x + 40, y: n.position.y + 40 } }; });
    set((s) => ({ nodes: [...s.nodes.map((n) => ({ ...n, selected: false })), ...fresh], edges: [...s.edges, ...s.edges.filter((e) => map[e.source] && map[e.target]).map((e) => ({ ...e, id: `e${uid++}`, source: map[e.source], target: map[e.target] }))] }));
    // edges between copied nodes (original edges are in s.edges)
  },
  duplicateSelected() { get().copy(); get().paste(); },
  autoLayout() {
    get().commit();
    const { nodes, edges } = get(); const depth: Record<string, number> = {};
    const d = (id: string, seen = new Set<string>()): number => { if (depth[id] != null) return depth[id]; if (seen.has(id)) return 0; seen.add(id); const ps = edges.filter((e) => e.target === id).map((e) => d(e.source, seen)); return (depth[id] = ps.length ? Math.max(...ps) + 1 : 0); };
    nodes.forEach((n) => d(n.id)); const rows: Record<number, number> = {};
    set({ nodes: nodes.map((n) => { const c = depth[n.id]; const r = rows[c] ?? 0; rows[c] = r + 1; return { ...n, position: { x: 60 + c * 290, y: 60 + r * 170 } }; }) });
  },
  loadTemplate(i) {
    const t = TEMPLATES[i], specs = get().specs; uid = 1;
    const nodes: RFNode[] = t.nodes.map(([id, type, p, x, y]) => ({ id, type: "card", position: { x, y: y + 80 }, data: { type, params: { ...specs[type].params.reduce((a, q) => ({ ...a, [q.name]: q.default }), {}), ...p } } }));
    const edges: Edge[] = t.edges.map(([s, d], k) => {
      const sn = nodes.find((n) => n.id === s)!, dn = nodes.find((n) => n.id === d)!; const so = specs[sn.data.type].outputs[0];
      const di = specs[dn.data.type].inputs.find((p) => p.type === so.type)!; return { id: `t${k}`, source: s, target: d, sourceHandle: so.name, targetHandle: di.name };
    });
    set({ nodes, edges, projectName: t.name.replace(/ \(Template \d+\)/, ""), job: null, issues: [], selectedId: null, past: [], future: [], dirty: false });
  },
  newProject() { set({ nodes: [], edges: [], projectName: "Untitled project", job: null, issues: [], selectedId: null, past: [], future: [], journal: {}, dirty: false }); },
  setName(n) { set({ projectName: n, dirty: true }); }, setMode(m) { set({ mode: m }); }, setTab(t) { set({ bottomTab: t }); }, setBottomH(h) { set({ bottomH: Math.max(120, Math.min(700, h)) }); },
  setJournal(k, v) { set((s) => ({ journal: { ...s.journal, [k]: v }, dirty: true })); },
  workflow() { const s = get(); return { name: s.projectName, journal: s.journal, nodes: s.nodes.map((n) => ({ id: n.id, type: n.data.type, params: n.data.params, position: n.position })), edges: s.edges.map((e) => ({ id: e.id, source: e.source, target: e.target, sourceHandle: e.sourceHandle, targetHandle: e.targetHandle })) }; },
  async validate() { try { const r = await api.validate(get().workflow()); set({ issues: r.issues }); get().toast(r.valid ? "Workflow is valid" : `${r.issues.filter((i: Issue) => i.level === "error").length} problem(s) found`, r.valid ? "ok" : "error"); if (!r.valid) get().setTab("errors"); return r.valid; } catch (e: any) { get().toast(e.message, "error"); return false; } },
  async run(target) {
    if (poller) clearInterval(poller);
    try {
      set({ job: null });
      const { job_id } = await api.run(get().workflow(), target);
      get().setTab("logs");
      const tick = async () => { try { const j = await api.job(job_id); set({ job: j, issues: j.issues || [] }); if (j.status === "Completed" || j.status === "Failed") { clearInterval(poller); poller = null; if (j.status === "Failed") { get().toast("Run failed — see Errors", "error"); get().setTab("errors"); } else get().toast("Run completed", "ok"); } } catch (e: any) { clearInterval(poller); get().toast(e.message, "error"); } };
      await tick(); poller = setInterval(tick, 450);
    } catch (e: any) { get().toast(e.message, "error"); }
  },
  async cancel() { const j = get().job; if (j && !["Completed", "Failed"].includes(j.status)) { await api.cancel(j.id); get().toast("Stopping after the current node…"); } },
  exportProject() {
    const s = get(); const wf = s.workflow();
    const dsRefs = s.nodes.filter((n) => n.data.type === "data_input" && n.data.params.source === "upload" && n.data.params.dataset_id).map((n) => { const m = s.datasets.find((d) => d.id === n.data.params.dataset_id); return { id: n.data.params.dataset_id, filename: m?.filename ?? "unknown", rows: m?.rows, cols: m?.cols }; });
    return { format: "visual-ml-studio-project", version: 1, name: s.projectName, saved_at: new Date().toISOString(), settings: { mode: s.mode }, journal: s.journal, datasets: dsRefs,
      nodes: wf.nodes.map((n: any) => ({ id: n.id, type: "card", position: n.position, data: { type: n.type, params: n.params } })), edges: wf.edges,
      results: s.job ? { job_id: s.job.id, status: s.job.status, metrics: Object.fromEntries(Object.entries(s.job.nodes).filter(([, v]) => v.preview?.kind === "metrics").map(([k, v]) => [k, v.preview.metrics])) } : null };
  },
  async importProject(p) {
    if (p.format !== "visual-ml-studio-project") throw new Error("This file is not a Visual ML Studio project.");
    const chk = await api.checkProject(p);
    set({ nodes: p.nodes, edges: p.edges, projectName: p.name, journal: p.journal || {}, mode: p.settings?.mode || "beginner", job: null, issues: [], past: [], future: [], selectedId: null, dirty: false });
    if (chk.missing.length) get().toast(`${chk.missing.length} dataset file(s) missing: ${chk.missing.map((m: any) => m.filename).join(", ")}. Re-upload the same file with 'Upload Dataset' to reconnect.`, "error");
    else get().toast(`Opened '${p.name}'`, "ok");
  },
}));
