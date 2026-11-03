import { useEffect, useRef, useState } from "react";
import { useStore } from "../store";
import { api } from "../api";
import { TEMPLATES } from "../templates";
import { fmt } from "./common";

function Modal({ title, onClose, children, wide }: any) {
  useEffect(() => { const k = (e: KeyboardEvent) => e.key === "Escape" && onClose(); window.addEventListener("keydown", k); return () => window.removeEventListener("keydown", k); }, []);
  return <div className="fixed inset-0 bg-black/35 z-50 flex items-center justify-center p-4" onMouseDown={(e) => e.target === e.currentTarget && onClose()} role="dialog" aria-modal="true" aria-label={title}>
    <div className={`bg-white rounded-lg shadow-xl max-h-[88vh] overflow-auto ${wide ? "w-[900px]" : "w-[520px]"}`}><div className="flex items-center px-4 py-2 border-b border-line"><b className="flex-1 text-[14px]">{title}</b><button className="btn" onClick={onClose}>Close</button></div><div className="p-4">{children}</div></div></div>;
}

function UploadModal({ onClose }: any) {
  const [busy, setBusy] = useState(false); const [res, setRes] = useState<any>(null); const [err, setErr] = useState(""); const toast = useStore((s) => s.toast); const refresh = useStore((s) => s.refreshDatasets); const addDuplicateRemovalStep = useStore((s) => s.addDuplicateRemovalStep);
  const addNode = useStore((s) => s.addNode); const nodes = useStore((s) => s.nodes); const setParams = useStore((s) => s.setParams); const [list, setList] = useState<any[]>([]); const [builtin, setBuiltin] = useState<any[]>([]);
  useEffect(() => { api.datasets().then(setList); api.builtin().then(setBuiltin); }, [res]);
  const up = async (f: File) => { setBusy(true); setErr(""); try { const r = await api.upload(f); setRes(r); refresh(); } catch (e: any) { setErr(e.message); } finally { setBusy(false); } };
  const use = (id: string) => { const existing = nodes.find((n) => n.data.type === "data_input"); if (existing) { setParams(existing.id, { source: "upload", dataset_id: id }); toast("Dataset connected to the existing Dataset Input node", "ok"); } else { addNode("data_input", { x: 60, y: 120 }); setTimeout(() => { const n = useStore.getState().nodes.at(-1)!; useStore.getState().setParams(n.id, { source: "upload", dataset_id: id }); }, 0); toast("Dataset Input node added", "ok"); } onClose(); };
  const p = res?.profile;
  return (<Modal title="Upload dataset" onClose={onClose} wide>
    <div className="border-2 border-dashed border-line rounded-lg p-6 text-center" onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); e.dataTransfer.files[0] && up(e.dataTransfer.files[0]); }}>
      <p className="mb-2">Drop a file here or choose one. Supported: CSV, TSV, Excel (.xlsx), Parquet, JSON, JSON Lines. Files stay on your machine.</p>
      <input type="file" accept=".csv,.tsv,.xlsx,.parquet,.json,.jsonl" onChange={(e) => e.target.files?.[0] && up(e.target.files[0])} aria-label="Choose dataset file" />{busy && <div className="mt-2">Reading and profiling…</div>}</div>
    {err && <div className="mt-2 text-red-700 bg-red-50 p-2 rounded">{err}</div>}
    {p && (<div className="mt-3 text-xs space-y-2"><div className="flex items-center gap-2 flex-wrap"><b className="text-sm">{res.meta.filename}</b><span>{p.rows} rows × {p.cols} columns · {p.duplicates} duplicates · {p.memory_mb} MB</span><button className="btn btn-primary ml-auto" onClick={() => use(res.meta.id)}>Use in workflow</button>{p.duplicates > 0 && <button className="btn" onClick={addDuplicateRemovalStep}>Remove duplicate rows in workflow</button>}</div>
      {p.warnings.length > 0 && <ul className="text-amber-700 list-disc pl-4">{p.warnings.map((w: string, i: number) => <li key={i}>{w}</li>)}</ul>}
      <div>Possible targets: <b>{p.possible_targets.join(", ")}</b> · Numerical: {p.numerical.length} · Categorical: {p.categorical.length}</div>
      <div className="overflow-auto max-h-56 border border-line rounded"><table className="tbl w-full"><thead><tr><th>Column</th><th>dtype</th><th>Inferred</th><th>Missing</th><th>Unique</th><th>mean</th><th>min</th><th>max</th></tr></thead><tbody>
        {p.columns.map((c: any) => { const s = p.stats[c.name] || {}; return <tr key={c.name}><td>{c.name}</td><td>{c.dtype}</td><td>{c.kind}</td><td>{c.missing} ({c.missing_pct}%)</td><td>{c.unique}</td><td>{fmt(s.mean)}</td><td>{fmt(s.min)}</td><td>{fmt(s.max)}</td></tr>; })}</tbody></table></div>
      <div className="overflow-auto max-h-40 border border-line rounded"><table className="tbl mono"><thead><tr>{p.preview.columns.map((c: string) => <th key={c}>{c}</th>)}</tr></thead><tbody>{p.preview.rows.slice(0, 10).map((r: any[], i: number) => <tr key={i}>{r.map((v, j) => <td key={j}>{v == null ? "∅" : String(v)}</td>)}</tr>)}</tbody></table></div></div>)}
    {list.length > 0 && <div className="mt-4"><b>Previously uploaded</b><ul className="mt-1 space-y-1">{list.map((d) => <li key={d.id} className="flex items-center gap-2 text-xs"><span className="flex-1">{d.filename} · {d.rows}×{d.cols}</span><button className="btn" onClick={() => use(d.id)}>Use</button></li>)}</ul></div>}
    <div className="mt-4 text-xs"><b>Built-in educational datasets</b> (add a Dataset Input node and pick one):<ul className="mt-1 list-disc pl-4 text-mute">{builtin.map((b) => <li key={b.key}>{b.label} — {b.task}; {b.source}. License: {b.license}</li>)}</ul></div>
  </Modal>);
}

const HELP = [["Build", "Drag components from the left library onto the canvas (or double-click). Connect an output port on the right of a node to an input port on the left of another. Port colors show data types; invalid connections are refused with an explanation."],
  ["Typical flow", "Dataset Input → Select Features & Target → Train/Test Split → (Imputer → Scaler/Encoder) → model → Model Evaluation → Model Chart. Preprocessing nodes attach to the split and are fitted on training rows only, so there is no leakage."],
  ["Compare models", "Branch several models from the same split node (or the same preprocessing node), add an Evaluation after each and connect them to Compare Models. Different splits trigger a fairness warning."],
  ["Run", "Run Workflow executes everything; “Run up to this node” runs only the selected node and its dependencies. Unchanged nodes are served from cache; editing a node invalidates only what depends on it."],
  ["Export", "Generated Code shows runnable Python. Download .py or a Jupyter notebook that does not depend on this app. Save Project writes a versioned JSON file that references datasets by content hash."],
  ["Shortcuts", "Ctrl+Z / Ctrl+Y undo/redo · Ctrl+C / V / D copy, paste, duplicate · Delete remove · Ctrl+Enter run · Ctrl+S save · Right-click a node for its menu · Shift-drag to box-select."],
  ["Modes", "Beginner hides advanced settings (cross-validation, hyperparameter search). Advanced exposes everything. Switching never changes your graph."]];

export default function Topbar() {
  const s = useStore(); const job = s.job; const running = !!job && !["Completed", "Failed"].includes(job.status); const [modal, setModal] = useState<"" | "upload" | "help" | "settings" | "open" | "history">(""); const fileRef = useRef<HTMLInputElement>(null);
  const [projects, setProjects] = useState<any[]>([]);
  const save = async () => { const p = s.exportProject(); const blob = new Blob([JSON.stringify(p, null, 2)], { type: "application/json" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `${s.projectName.replace(/[^A-Za-z0-9_-]+/g, "_")}.vmls.json`; a.click(); try { await api.saveProject(p); } catch {} s.toast("Project saved (downloaded + stored on the server)", "ok"); };
  useEffect(() => { const h = () => save(); window.addEventListener("vmls-save", h); return () => window.removeEventListener("vmls-save", h); });
  const statusColor = job?.status === "Completed" ? "#15803d" : job?.status === "Failed" ? "#b91c1c" : running ? "#2563a8" : "#6b7280";
  return (<header className="h-12 bg-white border-b border-line flex items-center gap-2 px-3 shrink-0 overflow-x-auto">
    <div className="font-bold text-[15px] mr-1 whitespace-nowrap" style={{ color: "#0f766e" }}>🧪 Visual ML Studio</div>
    <input className="inp !w-44 font-medium" value={s.projectName} onChange={(e) => s.setName(e.target.value)} aria-label="Project name" />
    <button className="btn" onClick={() => { if (!s.dirty || confirm("Discard the current canvas?")) s.newProject(); }}>New</button>
    <button className="btn" onClick={() => { api.projects().then(setProjects); setModal("open"); }}>Open</button><button className="btn" onClick={save}>Save</button>
    <button className="btn" onClick={() => setModal("upload")}>Upload Dataset</button>
    <select className="inp !w-40" aria-label="Load template" value="" onChange={(e) => e.target.value !== "" && s.loadTemplate(+e.target.value)}><option value="">Templates…</option>{TEMPLATES.map((t, i) => <option key={i} value={i} title={t.description}>{t.name}</option>)}</select>
    <span className="w-px h-6 bg-line" />
    <button className="btn" onClick={s.validate}>Validate</button>
    <button className="btn btn-primary" disabled={running || !s.nodes.length} onClick={() => s.run()}>▶ Run Workflow</button>
    <button className="btn btn-danger" disabled={!running} onClick={s.cancel}>■ Stop</button>
    <span className="px-2 py-0.5 rounded-full text-white text-xs whitespace-nowrap" style={{ background: statusColor }} aria-live="polite">{job ? job.phase : "Idle"}</span>
    <span className="w-px h-6 bg-line" />
    <button className="btn" onClick={() => s.setTab("code")}>Generate Python</button>
    <button className="btn" onClick={() => api.download("notebook", s.workflow(), s.projectName).then(() => s.toast("Notebook exported and validated", "ok")).catch((e) => s.toast(e.message, "error"))}>Export Notebook</button>
    <button className="btn" onClick={() => s.setTab("compare")}>Experiment History</button>
    <div className="ml-auto flex items-center gap-2"><div className="flex border border-line rounded-md overflow-hidden" role="group" aria-label="Interface mode">{(["beginner", "advanced"] as const).map((m) => <button key={m} className={`px-2 py-1 ${s.mode === m ? "bg-accent text-white" : "bg-white"}`} onClick={() => s.setMode(m)} aria-pressed={s.mode === m}>{m === "beginner" ? "Beginner" : "Advanced"}</button>)}</div>
      <button className="btn" onClick={() => setModal("settings")}>Settings</button><button className="btn" onClick={() => setModal("help")}>Help</button></div>
    {modal === "upload" && <UploadModal onClose={() => setModal("")} />}
    {modal === "help" && <Modal title="Help and documentation" onClose={() => setModal("")}>{HELP.map(([h, t]) => <div key={h} className="mb-3"><div className="font-semibold">{h}</div><p className="text-mute">{t}</p></div>)}<p className="text-mute text-xs">Every node has a Learn tab in the right panel explaining what it does, when to use it and its pitfalls.</p></Modal>}
    {modal === "settings" && <Modal title="Settings" onClose={() => setModal("")}><p className="mb-2">Interface mode: <b>{s.mode}</b> (use the toggle in the top bar).</p><button className="btn" onClick={() => api.clearHistory().then(() => s.toast("History cleared", "ok"))}>Clear experiment history</button>
      <p className="text-mute text-xs mt-3">Server limits (upload size, job timeout) are set with the environment variables VMLS_MAX_UPLOAD_MB and VMLS_JOB_TIMEOUT_S.</p></Modal>}
    {modal === "open" && <Modal title="Open project" onClose={() => setModal("")}><input ref={fileRef} type="file" accept=".json" className="hidden" onChange={async (e) => { const f = e.target.files?.[0]; if (!f) return; try { await s.importProject(JSON.parse(await f.text())); setModal(""); } catch (er: any) { s.toast(er.message, "error"); } }} />
      <button className="btn btn-primary" onClick={() => fileRef.current?.click()}>Open a project file…</button><div className="mt-3 font-semibold">Saved on this server</div>
      {projects.length ? <ul className="mt-1 space-y-1">{projects.map((p) => <li key={p.slug} className="flex items-center gap-2"><span className="flex-1">{p.name} <span className="text-mute text-xs">{p.nodes} nodes</span></span><button className="btn" onClick={async () => { try { await s.importProject(await api.loadProject(p.slug)); setModal(""); } catch (er: any) { s.toast(er.message, "error"); } }}>Open</button></li>)}</ul> : <p className="text-mute">No saved projects yet.</p>}</Modal>}
  </header>);
}
