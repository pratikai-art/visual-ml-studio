import { useEffect, useMemo, useRef, useState } from "react";
import hljs from "highlight.js/lib/core"; import python from "highlight.js/lib/languages/python";
import { useStore } from "../store";
import { api } from "../api";
import { DataTable, PlotlyChart, fmt, STATUS } from "./common";
hljs.registerLanguage("python", python);

const TABS = [["logs", "Execution Logs"], ["data", "Data Preview"], ["metrics", "Metrics"], ["charts", "Charts"], ["predictions", "Predictions"], ["errors", "Errors & Warnings"], ["code", "Generated Code"], ["compare", "Experiment Comparison"], ["journal", "Journal"]];

export default function BottomPanel() {
  const tab = useStore((s) => s.bottomTab), setTab = useStore((s) => s.setTab), h = useStore((s) => s.bottomH), setH = useStore((s) => s.setBottomH);
  const job = useStore((s) => s.job); const issues = useStore((s) => s.issues);
  const nErr = (job ? Object.values(job.nodes).filter((n) => n.error).length : 0) + issues.filter((i) => i.level === "error").length;
  const drag = useRef<{ y: number; h: number } | null>(null);
  useEffect(() => { const mv = (e: MouseEvent) => drag.current && setH(drag.current.h + (drag.current.y - e.clientY)); const up = () => (drag.current = null); window.addEventListener("mousemove", mv); window.addEventListener("mouseup", up); return () => { window.removeEventListener("mousemove", mv); window.removeEventListener("mouseup", up); }; }, []);
  return (
    <section className="bg-white border-t border-line flex flex-col shrink-0" style={{ height: h }} aria-label="Results panel">
      <div className="h-1.5 cursor-row-resize bg-line/60 hover:bg-accent/40" onMouseDown={(e) => (drag.current = { y: e.clientY, h })} role="separator" aria-label="Resize panel" />
      <div className="flex border-b border-line overflow-x-auto" role="tablist">{TABS.map(([k, l]) => <button key={k} role="tab" aria-selected={tab === k} className={`px-3 py-1.5 whitespace-nowrap font-medium ${tab === k ? "border-b-2 border-accent text-accent" : "text-mute"}`} onClick={() => setTab(k)}>{l}{k === "errors" && nErr > 0 ? <span className="ml-1 bg-red-600 text-white rounded-full px-1.5 text-[10px]">{nErr}</span> : null}</button>)}</div>
      <div className="flex-1 min-h-0 overflow-auto p-2">
        {tab === "logs" && <Logs />}{tab === "data" && <DataPreview />}{tab === "metrics" && <Metrics />}{tab === "charts" && <Charts />}{tab === "predictions" && <Predictions />}
        {tab === "errors" && <Errors />}{tab === "code" && <Code />}{tab === "compare" && <Compare />}{tab === "journal" && <Journal />}
      </div></section>);
}

const useSel = () => { const id = useStore((s) => s.selectedId); const job = useStore((s) => s.job); const nodes = useStore((s) => s.nodes); const specs = useStore((s) => s.specs); return { id, job, nodes, specs, st: id ? job?.nodes[id] : undefined }; };
const nameOf = (nodes: any[], specs: any, id: string) => specs[nodes.find((n) => n.id === id)?.data.type]?.name ?? id;
const Empty = ({ children }: any) => <div className="text-mute p-2">{children}</div>;

function Logs() {
  const job = useStore((s) => s.job); const nodes = useStore((s) => s.nodes); const specs = useStore((s) => s.specs); const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [job?.logs.length]);
  if (!job) return <Empty>Press <b>Run Workflow</b> to execute. Real training logs appear here as each node runs.</Empty>;
  const pct = Math.round((job.progress || 0) * 100);
  return (<div><div className="flex items-center gap-3 mb-2"><span className="font-semibold">{job.phase}</span><div className="flex-1 h-2 bg-paper rounded overflow-hidden border border-line"><div className="h-full bg-accent transition-all" style={{ width: `${job.status === "Completed" ? 100 : pct}%` }} /></div><span className="mono">{job.status === "Completed" ? 100 : pct}%</span></div>
    <div className="mono text-xs space-y-0.5">{job.logs.map((l, i) => <div key={i} className={/ERROR|error/.test(l.text) ? "text-red-700" : ""}>{l.node_id ? <span className="text-mute">[{nameOf(nodes, specs, l.node_id)}] </span> : null}{l.text.replace(/^\[\d\d:\d\d:\d\d\] /, "")}</div>)}<div ref={end} /></div></div>);
}

function DataPreview() {
  const { st, id } = useSel(); const pv = st?.preview;
  if (!id) return <Empty>Select a node, then run the workflow to see the data at that point.</Empty>;
  if (!pv) return <Empty>No output yet for this node. Run the workflow (or “Run up to this node”).</Empty>;
  const table = pv.table || pv.train_preview;
  return (<div className="space-y-3">
    <div className="flex gap-4 flex-wrap text-xs text-mute">{pv.shape && <span>Shape: <b className="text-ink">{pv.shape[0]} × {pv.shape[1]}</b></span>}{pv.name && <span>Source: <b className="text-ink">{pv.name}</b></span>}{pv.license && <span>License: {pv.license}</span>}
      {pv.excluded && <span>Excluded columns: <b className="text-ink">{pv.excluded.join(", ") || "none"}</b></span>}{pv.dropped && <span>Dropped: <b className="text-ink">{pv.dropped.join(", ") || "none"}</b></span>}
      {pv.kind === "preprocess" && <span>Missing {pv.missing_before} → {pv.missing_after}; columns {pv.columns_before.length} → {pv.columns_after.length} (fitted on training rows only)</span>}
      {pv.kind === "split" && <span>{pv.n_train} train / {pv.n_test} test · seed {pv.seed} · shuffle {String(pv.shuffle)} · stratified {String(pv.stratified)}</span>}</div>
    {pv.kind === "preprocess" && <div className="text-xs"><b>Before:</b> {pv.columns_before.join(", ")}<br /><b>After:</b> {pv.columns_after.join(", ")}</div>}
    {pv.class_distribution && <ClassDist d={pv.class_distribution} />}
    {pv.kind === "profile" && <Profile p={pv.profile} />}{pv.kind === "missing" && <MissingTable rows={pv.missing_table} />}
    {pv.n_duplicates != null && <div className="text-xs">{pv.duplicates_removed ? <>Removed <b>{pv.n_duplicates}</b> duplicate rows.</> : <>Duplicate rows found: <b>{pv.n_duplicates}</b></>}{pv.duplicates_preview?.rows.length > 0 && <DataTable table={pv.duplicates_preview} maxHeight={120} />}</div>}
    {pv.kind === "clustering" && <div className="text-xs"><div className="mono">{Object.entries(pv.metrics).map(([k, v]) => `${k}: ${fmt(v)}`).join("   ")}</div><div className="text-mute">{pv.note}</div></div>}
    {pv.kind === "pca" && <div className="text-xs mono">Explained variance: {pv.explained_variance.map((v: number, i: number) => `PC${i + 1} ${(v * 100).toFixed(1)}%`).join(" · ")} (cumulative {(pv.cumulative.at(-1) * 100).toFixed(1)}%)</div>}
    {pv.kind === "model" && <div className="text-xs"><b>Features used by the model ({pv.n_features}):</b> {pv.features.join(", ")}</div>}
    {table && <DataTable table={table} maxHeight={Math.max(120, useStore.getState().bottomH - 200)} />}
    {pv.kind === "model" && pv.search && <Search s={pv.search} />}
  </div>);
}
const ClassDist = ({ d }: any) => { const tr = d.train ? d : { all: d }; return <div className="text-xs mono">{Object.entries(tr).map(([k, v]: any) => <div key={k}><b>{k}</b>: {Object.entries(v).map(([c, n]) => `${c}=${n}`).join("  ")}</div>)}</div>; };
const MissingTable = ({ rows }: any) => <table className="tbl text-xs"><thead><tr><th>Column</th><th>Missing</th><th>%</th></tr></thead><tbody>{rows.map((r: any) => <tr key={r.column}><td>{r.column}</td><td>{r.missing}</td><td>{r.percent}</td></tr>)}</tbody></table>;
function Profile({ p }: any) {
  return (<div className="text-xs space-y-2"><div>{p.rows} rows × {p.cols} columns · {p.duplicates} duplicate rows · {p.memory_mb} MB</div>
    {p.warnings.length > 0 && <ul className="text-amber-700 list-disc pl-4">{p.warnings.map((w: string, i: number) => <li key={i}>{w}</li>)}</ul>}
    <table className="tbl"><thead><tr><th>Column</th><th>Type</th><th>Inferred</th><th>Missing</th><th>Unique</th><th>mean</th><th>std</th><th>min</th><th>max</th></tr></thead><tbody>
      {p.columns.map((c: any) => { const s = p.stats[c.name] || {}; return <tr key={c.name}><td>{c.name}</td><td>{c.dtype}</td><td>{c.kind}</td><td>{c.missing} ({c.missing_pct}%)</td><td>{c.unique}</td><td>{fmt(s.mean)}</td><td>{fmt(s.std)}</td><td>{fmt(s.min)}</td><td>{fmt(s.max)}</td></tr>; })}</tbody></table>
    <div className="text-mute">Possible targets: {p.possible_targets.join(", ")}</div></div>);
}
function Search({ s }: any) {
  return (<div className="text-xs"><b>Hyperparameter search</b> — {s.n_trials} trials, scoring {s.scoring}, best CV score {fmt(s.best_score)}; best parameters: <span className="mono">{JSON.stringify(s.best_params)}</span>
    <table className="tbl mt-1"><thead><tr><th>Rank</th><th>Parameters</th><th>Mean CV</th><th>Std</th></tr></thead><tbody>{s.trials.map((t: any, i: number) => <tr key={i}><td>{t.rank}</td><td className="mono">{JSON.stringify(t.params)}</td><td>{fmt(t.mean)}</td><td>{fmt(t.std)}</td></tr>)}</tbody></table></div>);
}

const HIGHER = (k: string) => !["mae", "mse", "rmse", "median_ae", "mape", "log_loss", "davies_bouldin"].includes(k);
function Metrics() {
  const { job, nodes, specs, st } = useSel(); const all = job ? Object.entries(job.nodes).filter(([, v]) => v.preview?.kind === "metrics") : [];
  const list = st?.preview?.kind === "metrics" ? [[useStore.getState().selectedId!, st]] : all;
  if (!list.length) return <Empty>Add a <b>Model Evaluation</b> node after a trained model and run the workflow.</Empty>;
  return (<div className="space-y-4">{list.map(([id, n]: any) => { const p = n.preview; return (<div key={id} className="border border-line rounded p-2">
    <div className="font-semibold">{p.model} <span className="text-mute font-normal">· {p.task} · {p.n_train} train / {p.n_test} test rows · fit {p.train_time.toFixed(2)}s · predict {(p.infer_time * 1000).toFixed(1)}ms</span></div>
    <table className="tbl text-xs mt-1"><thead><tr><th>Metric</th><th>Test (held-out)</th><th>Train</th></tr></thead><tbody>{Object.entries(p.metrics).map(([k, v]) => <tr key={k}><td>{k}</td><td className="mono font-semibold">{fmt(v)}</td><td className="mono text-mute">{fmt(p.train_metrics?.[k])}</td></tr>)}</tbody></table>
    {p.cv && <div className="text-xs mt-1">Cross-validation ({p.cv.scoring}): <b className="mono">{fmt(p.cv.mean)}{p.cv.std != null ? ` ± ${fmt(p.cv.std)}` : ""}</b> {p.cv.note ? `(${p.cv.note})` : ""}</div>}
    {n.warnings.length > 0 && <ul className="text-xs text-amber-700 list-disc pl-4 mt-1">{n.warnings.map((w: string, i: number) => <li key={i}>{w}</li>)}</ul>}
    <div className="mt-2 bg-paper rounded p-2"><div className="font-semibold text-xs">Why did this result happen?</div><ul className="text-xs list-disc pl-4 space-y-0.5 mt-1">{p.insights.map((t: string, i: number) => <li key={i}>{t}</li>)}</ul>
      <div className="text-[11px] text-mute mt-1">Generated from your actual metrics, split sizes and class balance — not from a generic template.</div></div></div>); })}</div>);
}

function Charts() {
  const { job, nodes, specs, st, id } = useSel();
  const figs = id && st?.preview?.figure ? [[id, st]] : job ? Object.entries(job.nodes).filter(([, v]) => v.preview?.kind === "figure") : [];
  if (!figs.length) return <Empty>Add a <b>Dataset Chart</b> or <b>Model Chart</b> node and run. Select a chart node to view only that chart.</Empty>;
  return <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">{figs.map(([k, v]: any) => <div key={k} className="border border-line rounded"><PlotlyChart fig={v.preview.figure} height={Math.max(300, useStore.getState().bottomH - 70)} /></div>)}</div>;
}

function Predictions() {
  const { job, st, id } = useSel(); const cand = id && st?.preview?.predictions ? st : job ? Object.values(job.nodes).find((v) => v.preview?.predictions) : null;
  if (!cand) return <Empty>Select a trained model node after a run to see test-set predictions.</Empty>;
  return <div><div className="text-xs text-mute mb-1">{cand.preview.algorithm}: predictions on the held-out test rows.</div><DataTable table={cand.preview.predictions} maxHeight={Math.max(120, useStore.getState().bottomH - 130)} /></div>;
}

function Errors() {
  const { job, nodes, specs } = useSel(); const issues = useStore((s) => s.issues); const select = useStore((s) => s.select);
  const errs = job ? Object.entries(job.nodes).filter(([, v]) => v.error || v.warnings.length) : [];
  if (!errs.length && !issues.length) return <Empty>No errors or warnings. Use <b>Validate</b> to check the graph before running.</Empty>;
  return (<div className="space-y-2 text-xs">{issues.map((i, k) => <div key={"i" + k} className={`rounded p-2 ${i.level === "error" ? "bg-red-50 text-red-800" : "bg-amber-50 text-amber-800"}`}>{i.node_id && <button className="underline mr-1" onClick={() => select(i.node_id!)}>{nameOf(nodes, specs, i.node_id)}</button>}{i.message}</div>)}
    {errs.map(([nid, v]) => (<div key={nid} className="border border-line rounded p-2"><button className="font-semibold underline" onClick={() => select(nid)}>{nameOf(nodes, specs, nid)}</button> <span style={{ color: STATUS[v.status]?.color }}>{v.status}</span>
      {v.error && <div className="text-red-800 mt-1">{v.error.message}{v.error.technical && <details className="mt-1"><summary className="cursor-pointer text-mute">Technical details</summary><pre className="mono whitespace-pre-wrap text-[11px] bg-paper p-2 rounded">{v.error.technical}</pre></details>}</div>}
      {v.warnings.map((w: string, i: number) => <div key={i} className="text-amber-700 mt-0.5">⚠ {w}</div>)}</div>))}</div>);
}

function Code() {
  const wf = useStore((s) => s.workflow); const id = useStore((s) => s.selectedId); const name = useStore((s) => s.projectName); const sig = useStore((s) => JSON.stringify(s.nodes.map((n) => n.data.params)) + s.edges.length + s.nodes.length);
  const [code, setCode] = useState(""); const [err, setErr] = useState(""); const [scope, setScope] = useState<"all" | "node">("all"); const toast = useStore((s) => s.toast);
  useEffect(() => { let dead = false; const t = setTimeout(() => api.code(wf(), scope === "node" && id ? id : undefined).then((r) => { if (!dead) { setCode(r.code); setErr(""); } }).catch((e) => !dead && (setErr(e.message), setCode(""))), 300); return () => { dead = true; clearTimeout(t); }; }, [sig, scope, id]);
  const html = useMemo(() => (code ? hljs.highlight(code, { language: "python" }).value : ""), [code]);
  return (<div className="h-full flex flex-col"><div className="flex gap-2 mb-1 items-center"><select className="inp !w-auto" value={scope} onChange={(e) => setScope(e.target.value as any)}><option value="all">Complete workflow</option><option value="node">Up to selected node</option></select>
    <button className="btn" disabled={!code} onClick={() => navigator.clipboard.writeText(code).then(() => toast("Code copied", "ok"))}>Copy</button><button className="btn" disabled={!code} onClick={() => api.download("script", wf(), name).catch((e) => toast(e.message, "error"))}>Download .py</button>
    <button className="btn" disabled={!code} onClick={() => api.download("notebook", wf(), name).catch((e) => toast(e.message, "error"))}>Download .ipynb</button><span className="text-mute text-xs">Runnable Python that mirrors the current graph. Task is resolved from your data when set to “auto”.</span></div>
    {err ? <div className="text-red-700 text-xs bg-red-50 p-2 rounded">{err}</div> : <pre className="mono text-xs bg-[#fbfbf9] border border-line rounded p-2 overflow-auto flex-1"><code dangerouslySetInnerHTML={{ __html: html }} /></pre>}</div>);
}

function Compare() {
  const { job } = useSel(); const [hist, setHist] = useState<any[]>([]); const [metric, setMetric] = useState("auto");
  useEffect(() => { api.history().then(setHist).catch(() => {}); }, [job?.status]);
  const cmp = job ? Object.values(job.nodes).find((v) => v.preview?.kind === "compare") : null;
  const rows = useMemo(() => hist.flatMap((r) => r.models.map((m: any) => ({ ...m, run: r.run_id, time: r.time, workflow: r.workflow }))), [hist]);
  const keys = useMemo(() => Array.from(new Set(rows.flatMap((r) => Object.keys(r.metrics || {})))), [rows]);
  const m = metric === "auto" ? keys.includes("r2") ? "r2" : keys[0] : metric; const sorted = [...rows].filter((r) => r.metrics?.[m] != null).sort((a, b) => (HIGHER(m) ? b.metrics[m] - a.metrics[m] : a.metrics[m] - b.metrics[m]));
  return (<div className="space-y-4 text-xs">
    {cmp && <div><div className="font-semibold text-sm">Current run — ranked by {cmp.preview.metric} ({cmp.preview.lower_is_better ? "lower" : "higher"} is better). Best: {cmp.preview.best}</div>
      {cmp.warnings.map((w, i) => <div key={i} className="text-amber-700">⚠ {w}</div>)}
      <div className="overflow-auto"><table className="tbl mono"><thead><tr>{Object.keys(cmp.preview.table[0]).filter((k) => k !== "node").map((k) => <th key={k}>{k}</th>)}</tr></thead><tbody>{cmp.preview.table.map((r: any, i: number) => <tr key={i} className={i === 0 ? "bg-green-50" : ""}>{Object.entries(r).filter(([k]) => k !== "node").map(([k, v]) => <td key={k}>{fmt(v)}</td>)}</tr>)}</tbody></table></div>
      <details className="mt-1 text-mute"><summary className="cursor-pointer">How to read these trade-offs</summary><p className="mt-1 max-w-3xl">Accuracy hides class imbalance. Precision asks “of what I flagged, how much was right?”, recall asks “of what existed, how much did I find?”, F1 balances the two, ROC-AUC measures ranking quality across thresholds, PR-AUC is more informative when positives are rare. For regression, MAE is the average miss in target units, RMSE punishes large misses, and R² is the share of variance explained (it can be negative). The top-ranked model on one split is not guaranteed to be best in general — compare training time, stability (CV ±) and fairness of the split too.</p></details>
      <button className="btn mt-1" onClick={() => { const blob = new Blob([JSON.stringify(cmp.preview.table, null, 2)], { type: "application/json" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "comparison.json"; a.click(); }}>Download results (JSON)</button></div>}
    <div><div className="flex items-center gap-2 mb-1"><span className="font-semibold text-sm">Experiment history</span><select className="inp !w-auto" value={metric} onChange={(e) => setMetric(e.target.value)}><option value="auto">rank by: auto</option>{keys.map((k) => <option key={k}>{k}</option>)}</select>
      <button className="btn" onClick={() => api.clearHistory().then(() => setHist([]))}>Clear history</button></div>
      {!sorted.length ? <Empty>Runs with an evaluation node are saved here so you can compare experiments over time.</Empty> :
        <div className="overflow-auto"><table className="tbl mono"><thead><tr><th>#</th><th>Model</th><th>{m}</th><th>Seed</th><th>Test rows</th><th>fit s</th><th>Key params</th><th>When</th></tr></thead><tbody>{sorted.slice(0, 60).map((r, i) => <tr key={i}><td>{i + 1}</td><td>{r.model}</td><td className="font-semibold">{fmt(r.metrics[m])}</td><td>{r.seed ?? "—"}</td><td>{r.n_test ?? "—"}</td><td>{r.train_time != null ? (+r.train_time).toFixed(2) : "—"}</td>
          <td className="max-w-[260px] truncate" title={JSON.stringify(r.params)}>{JSON.stringify(r.params)}</td><td>{new Date(r.time * 1000).toLocaleTimeString()}</td></tr>)}</tbody></table></div>}
      <div className="text-mute mt-1">Rows with different test-set sizes/seeds are not strictly comparable.</div></div></div>);
}

function Journal() {
  const j = useStore((s) => s.journal), set = useStore((s) => s.setJournal), job = useStore((s) => s.job);
  const f = [["description", "Experiment description"], ["hypothesis", "Hypothesis"], ["changes", "Changes made"], ["observations", "Observations"], ["conclusions", "Conclusions"]];
  const ms = job ? Object.values(job.nodes).filter((v) => v.preview?.kind === "metrics") : [];
  return (<div className="grid grid-cols-2 gap-3 text-xs">{f.map(([k, l]) => <label key={k} className="block"><span className="font-semibold">{l}</span><textarea className="inp mt-0.5" rows={3} value={j[k] || ""} onChange={(e) => set(k, e.target.value)} /></label>)}
    <div className="bg-paper rounded p-2"><div className="font-semibold">Hypothesis vs observed results</div>{ms.length ? ms.map((m, i) => <div key={i} className="mono mt-1">{m.preview.model}: {Object.entries(m.preview.metrics).slice(0, 4).map(([k, v]) => `${k} ${fmt(v)}`).join(", ")}</div>) : <div className="text-mute">Run the workflow to compare with your hypothesis.</div>}
      <div className="text-mute mt-2">The journal is saved in the project file and written into exported notebooks.</div></div></div>);
}
