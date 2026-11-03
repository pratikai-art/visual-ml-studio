import { useEffect, useMemo, useRef, useState } from "react";
// @ts-ignore
import Plotly from "plotly.js-dist-min";

export const CAT_COLOR: Record<string, string> = { "Data Input": "#2563a8", "Data Exploration": "#5b6b8c", "Data Cleaning": "#9a6b12", "Data Preprocessing": "#7a4fa3", "Feature Selection": "#b03a6e",
  "Data Splitting": "#0f766e", "Machine Learning": "#1f2937", Clustering: "#4d7c0f", "Dimensionality Reduction": "#0e7490", Evaluation: "#b45309", Visualization: "#be185d" };
export const CAT_GLYPH: Record<string, string> = { "Data Input": "▤", "Data Exploration": "◎", "Data Cleaning": "✂", "Data Preprocessing": "⚙", "Feature Selection": "◈", "Data Splitting": "⑃", "Machine Learning": "∴", Clustering: "◌", "Dimensionality Reduction": "⇲", Evaluation: "✓", Visualization: "▥" };
export const PORT_COLOR: Record<string, string> = { dataset: "#2563a8", split: "#0f766e", trained: "#1f2937", metrics: "#b45309", figure: "#be185d" };
export const STATUS: Record<string, { label: string; color: string }> = { not_run: { label: "Not run", color: "#8b929c" }, ready: { label: "Ready", color: "#2563a8" }, running: { label: "Running", color: "#2563a8" },
  succeeded: { label: "Succeeded", color: "#15803d" }, warning: { label: "Warning", color: "#b45309" }, failed: { label: "Failed", color: "#b91c1c" }, skipped: { label: "Skipped", color: "#8b929c" } };

export function PlotlyChart({ fig, height = 380 }: { fig: any; height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current; if (!el || !fig) return;
    Plotly.react(el, fig.data, { ...fig.layout, autosize: true, height }, { responsive: true, displaylogo: false, toImageButtonOptions: { format: "png" } });
    return () => { try { Plotly.purge(el); } catch {} };
  }, [fig, height]);
  return <div ref={ref} style={{ width: "100%", minHeight: height }} />;
}

export function DataTable({ table, maxHeight = 220 }: { table: { columns: string[]; rows: any[][]; dtypes?: string[]; n_rows?: number }; maxHeight?: number }) {
  const [q, setQ] = useState(""); const [sort, setSort] = useState<{ i: number; dir: 1 | -1 } | null>(null);
  const rows = useMemo(() => {
    let r = table.rows; if (q) r = r.filter((row) => row.some((v) => String(v ?? "").toLowerCase().includes(q.toLowerCase())));
    if (sort) r = [...r].sort((a, b) => { const x = a[sort.i], y = b[sort.i]; if (x == null) return 1; if (y == null) return -1; return (x < y ? -1 : x > y ? 1 : 0) * sort.dir; });
    return r;
  }, [table, q, sort]);
  return (<div className="flex flex-col h-full min-h-0">
    <div className="flex items-center gap-2 pb-1"><input className="inp !w-48" placeholder="Search rows…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search rows" />
      <span className="text-mute">{table.n_rows != null ? `${table.n_rows} rows total · showing ${rows.length}` : `${rows.length} rows`} · click a header to sort</span></div>
    <div className="overflow-auto border border-line rounded bg-white" style={{ maxHeight }}><table className="tbl w-full text-xs mono"><thead><tr>
      {table.columns.map((c, i) => <th key={i} onClick={() => setSort(sort?.i === i ? { i, dir: (sort.dir * -1) as 1 | -1 } : { i, dir: 1 })} title={table.dtypes?.[i]}>{c}{sort?.i === i ? (sort.dir === 1 ? " ▲" : " ▼") : ""}<div className="text-[10px] font-normal text-mute">{table.dtypes?.[i]}</div></th>)}
    </tr></thead><tbody>{rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j} className={v == null ? "text-red-400" : ""}>{v == null ? "∅" : typeof v === "number" ? +v.toPrecision(6) : String(v)}</td>)}</tr>)}</tbody></table></div></div>);
}
export const fmt = (v: any) => (typeof v === "number" ? (Math.abs(v) >= 1000 || (Math.abs(v) < 0.001 && v !== 0) ? v.toExponential(2) : +v.toFixed(4)) : String(v ?? "—"));
