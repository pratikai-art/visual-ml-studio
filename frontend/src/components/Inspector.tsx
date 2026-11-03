import { useEffect, useMemo, useState } from "react";
import { useStore } from "../store";
import { api } from "../api";
import type { ParamSpec } from "../types";
import { CAT_COLOR, PORT_COLOR } from "./common";

const DOC_LABELS: Record<string, string> = { what: "What it does", how: "How it works", when_to_use: "When to use it", when_not_to_use: "When not to use it", limitations: "Limitations & failure modes",
  preprocessing: "Preprocessing needs", metrics: "Suitable metrics", complexity: "Complexity", hyperparameters: "Hyperparameters" };

function visible(p: ParamSpec, v: Record<string, any>, advanced: boolean) {
  if (p.advanced && !advanced && p.type !== "column" && p.type !== "columns") return false;
  if (!p.visible_if) return true; return Object.entries(p.visible_if).every(([k, allowed]) => allowed.includes(v[k]));
}
function check(p: ParamSpec, v: any): string | null {
  if (p.type === "int" || p.type === "float") { if (v === "" || v == null) return p.nullable ? null : "Needs a number"; const n = Number(v); if (isNaN(n)) return "Not a number"; if (p.type === "int" && !Number.isInteger(n)) return "Must be a whole number"; if (p.min != null && n < p.min) return `Min ${p.min}`; if (p.max != null && n > p.max) return `Max ${p.max}`; }
  if (p.type === "select" && p.options && !p.options.includes(v)) return "Invalid option"; return null;
}

function ColumnPicker({ options, value, onChange, multi, dtypes, suggested }: { options: string[]; value: any; onChange: (v: any) => void; multi: boolean; dtypes: Record<string, string>; suggested?: string }) {
  const [q, setQ] = useState(""); const sel: string[] = multi ? value || [] : value ? [value] : [];
  const shown = options.filter((o) => o.toLowerCase().includes(q.toLowerCase()));
  const toggle = (o: string) => (multi ? onChange(sel.includes(o) ? sel.filter((x) => x !== o) : [...sel, o]) : onChange(sel[0] === o ? "" : o));
  const numeric = (o: string) => /float|int/.test(dtypes[o] || "");
  return (<div className="border border-line rounded bg-white">
    <input className="inp !border-0 !border-b !rounded-none" placeholder={options.length ? `Search ${options.length} columns…` : "Connect an upstream node to list columns"} value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search columns" />
    {multi && <div className="flex gap-1 p-1 text-[11px] flex-wrap"><button className="btn !py-0" onClick={() => onChange(options)}>Select all</button><button className="btn !py-0" onClick={() => onChange([])}>Deselect all</button>
      <button className="btn !py-0" onClick={() => onChange(options.filter(numeric))}>Numeric</button><button className="btn !py-0" onClick={() => onChange(options.filter((o) => !numeric(o)))}>Non-numeric</button></div>}
    {!multi && <div className="flex gap-1 p-1 text-[11px]"><button type="button" className="btn !py-0" onClick={() => onChange("")}>Clear target</button>{suggested && options.includes(suggested) && <button type="button" className="btn !py-0" onClick={() => onChange(suggested)}>Use suggested: {suggested}</button>}</div>}
    <div className="max-h-52 overflow-y-auto">{shown.map((o) => (<label key={o} className="flex items-center gap-2 px-2 py-1 hover:bg-paper cursor-pointer">
      <input type={multi ? "checkbox" : "radio"} checked={sel.includes(o)} onChange={() => toggle(o)} /><span className="truncate">{o}</span><span className="ml-auto text-[10px] text-mute">{dtypes[o]}</span></label>))}</div>
    {multi && <div className="text-[11px] text-mute px-2 py-0.5 border-t border-line">{sel.length ? `${sel.length} selected` : "none selected (node default applies)"}</div>}</div>);
}

export default function Inspector() {
  const id = useStore((s) => s.selectedId); const node = useStore((s) => s.nodes.find((n) => n.id === s.selectedId)); const specs = useStore((s) => s.specs);
  const mode = useStore((s) => s.mode); const setParams = useStore((s) => s.setParams); const st = useStore((s) => (s.selectedId ? s.job?.nodes[s.selectedId] : undefined));
  const sig = useStore((s) => JSON.stringify([s.nodes.map((n) => [n.id, n.data.params]), s.edges.map((e) => [e.source, e.target])]));
  const issues = useStore((s) => s.issues); const getWf = useStore((s) => s.workflow); const run = useStore((s) => s.run);
  const spec = node ? specs[node.data.type] : null; const [draft, setDraft] = useState<Record<string, any>>({}); const [tab, setTab] = useState<"settings" | "learn">("settings");
  const [cols, setCols] = useState<{ name: string; dtype: string }[]>([]); const [colMsg, setColMsg] = useState(""); const [suggestedTarget, setSuggestedTarget] = useState("");
  useEffect(() => { if (node) setDraft({ ...node.data.params }); }, [id, node?.data.params]);
  const needsCols = !!spec?.params.some((p) => p.type === "column" || p.type === "columns");
  useEffect(() => {
    if (!id || !needsCols) return; let dead = false; const t = setTimeout(async () => { try { const r = await api.columns(getWf(), id); if (!dead) { setCols(r.columns); setColMsg(r.message || ""); setSuggestedTarget(r.suggested_target || ""); if (spec?.type === "select_target" && !node.data.params.target && r.suggested_target) setDraft((d) => ({ ...d, target: r.suggested_target })); } } catch { } }, 250);
    return () => { dead = true; clearTimeout(t); };
  }, [id, needsCols, sig]);
  const dtypes = useMemo(() => Object.fromEntries(cols.map((c) => [c.name, c.dtype])), [cols]);
  if (!node || !spec) return <aside className="w-[310px] shrink-0 bg-white border-l border-line p-4 text-mute">Select a node to see its settings, documentation and a beginner-friendly example.</aside>;
  const dirty = JSON.stringify(draft) !== JSON.stringify(node.data.params); const v = draft; const adv = mode === "advanced";
  const serverIssues = issues.filter((i) => i.node_id === id);
  const best = st?.preview?.search?.best_params;
  return (
    <aside className="w-[310px] shrink-0 bg-white border-l border-line flex flex-col min-h-0" aria-label="Inspector">
      <div className="px-3 py-2 border-b border-line"><div className="font-semibold text-[14px]">{spec.name}</div><div className="text-mute text-xs">{spec.category}{spec.subcategory ? ` · ${spec.subcategory}` : ""}</div>
        <div className="flex gap-1 mt-1 flex-wrap">{spec.compatibility.accepts.map((t, i) => <span key={i} className="text-[10px] px-1.5 rounded-full text-white" style={{ background: PORT_COLOR[t] }}>in: {t}</span>)}{spec.compatibility.produces.map((t, i) => <span key={i} className="text-[10px] px-1.5 rounded-full text-white" style={{ background: PORT_COLOR[t] }}>out: {t}</span>)}</div></div>
      <div className="flex border-b border-line">{(["settings", "learn"] as const).map((t) => <button key={t} className={`flex-1 py-1.5 font-medium ${tab === t ? "border-b-2 border-accent text-accent" : "text-mute"}`} onClick={() => setTab(t)}>{t === "settings" ? "Settings" : "Learn"}</button>)}</div>
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {tab === "settings" ? (<>
          {spec.params.filter((p) => visible(p, v, adv)).map((p) => {
            const err = check(p, v[p.name]); const isDef = JSON.stringify(v[p.name]) === JSON.stringify(p.default);
            return (<div key={p.name}><div className="flex items-center"><label className="font-medium flex-1" title={p.description}>{p.label}{p.advanced && <span className="text-[10px] text-mute ml-1">advanced</span>}</label>
              {!isDef && <button className="text-[11px] text-accent" onClick={() => setDraft({ ...v, [p.name]: p.default })}>reset</button>}</div>
              {p.type === "bool" ? <label className="flex items-center gap-2"><input type="checkbox" checked={!!v[p.name]} onChange={(e) => setDraft({ ...v, [p.name]: e.target.checked })} />{p.description}</label>
                : p.type === "select" ? <select className="inp" value={v[p.name] ?? ""} onChange={(e) => setDraft({ ...v, [p.name]: e.target.value })}>{p.options!.map((o) => <option key={o}>{o}</option>)}</select>
                : p.type === "int" || p.type === "float" ? (<div className="flex gap-2 items-center"><input className="inp" type="number" step={p.step ?? (p.type === "int" ? 1 : "any")} value={v[p.name] ?? ""} onChange={(e) => setDraft({ ...v, [p.name]: e.target.value === "" ? null : Number(e.target.value) })} />
                  {p.min != null && p.max != null && p.max - p.min <= 1000 && v[p.name] != null && <input type="range" aria-label={p.label} className="w-24" min={p.min} max={p.max} step={p.step ?? (p.type === "int" ? 1 : (p.max - p.min) / 100)} value={v[p.name]} onChange={(e) => setDraft({ ...v, [p.name]: Number(e.target.value) })} />}</div>)
                : p.type === "column" || p.type === "columns" ? <ColumnPicker options={cols.map((c) => c.name)} dtypes={dtypes} suggested={suggestedTarget} multi={p.type === "columns"} value={v[p.name]} onChange={(x) => setDraft({ ...v, [p.name]: x })} />
                : p.type === "text" ? <textarea className="inp mono" rows={3} value={v[p.name] ?? ""} onChange={(e) => setDraft({ ...v, [p.name]: e.target.value })} />
                : p.name === "dataset_id" ? <DatasetSelect value={v[p.name]} onChange={(x: string) => setDraft({ ...v, [p.name]: x })} /> : <input className="inp" value={v[p.name] ?? ""} onChange={(e) => setDraft({ ...v, [p.name]: e.target.value })} />}
              {p.type !== "bool" && p.description && <div className="text-[11px] text-mute mt-0.5">{p.description}</div>}
              {err && <div className="text-[11px] text-red-700">{err}</div>}</div>);
          })}
          {(needsCols && colMsg) && <div className="text-[11px] text-amber-700">{colMsg}</div>}
          {!adv && spec.params.some((p) => p.advanced) && <div className="text-[11px] text-mute">Advanced settings are hidden in Beginner mode.</div>}
          {serverIssues.map((i, k) => <div key={k} className="text-xs text-red-700 bg-red-50 rounded p-1.5">{i.message}</div>)}
        </>) : (<>
          <p>{spec.description}</p>
          {Object.entries(spec.doc).map(([k, t]) => <div key={k}><div className="font-semibold">{DOC_LABELS[k] ?? k}</div><p className="text-[12px] text-mute leading-snug">{t}</p></div>)}
          {spec.example && <div><div className="font-semibold">Beginner example</div><p className="text-[12px] text-mute">{spec.example}</p></div>}
          <div><div className="font-semibold">Compatibility</div><p className="text-[12px] text-mute">Accepts: {spec.compatibility.accepts.join(", ") || "nothing (a starting node)"} · Produces: {spec.compatibility.produces.join(", ") || "results shown in the bottom panel"}</p></div>
        </>)}
      </div>
      {tab === "settings" && (<div className="p-2 border-t border-line flex gap-2 flex-wrap">
        <button className="btn btn-primary flex-1" disabled={!dirty} onClick={() => setParams(node.id, draft, true)}>Apply configuration</button>
        <button className="btn" onClick={() => setDraft(spec.params.reduce((a, p) => ({ ...a, [p.name]: p.default }), {}))}>Reset all</button>
        <button className="btn w-full" onClick={() => run(node.id)}>Run up to this node</button>
        {best && <button className="btn w-full" onClick={() => { const patch: any = {}; Object.entries(best).forEach(([k, val]) => { if (spec.params.some((p) => p.name === k)) patch[k] = val; }); setParams(node.id, { ...patch, search: "none" }); }}>Apply best parameters &amp; turn search off</button>}
      </div>)}
    </aside>);
}

function DatasetSelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const ds = useStore((s) => s.datasets);
  return <select className="inp" value={value} onChange={(e) => onChange(e.target.value)}><option value="">— choose an uploaded dataset —</option>{ds.map((d) => <option key={d.id} value={d.id}>{d.filename} ({d.rows}×{d.cols})</option>)}
    {value && !ds.find((d) => d.id === value) && <option value={value}>{value} (missing — re-upload the file)</option>}</select>;
}
