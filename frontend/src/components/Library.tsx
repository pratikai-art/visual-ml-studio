import { useMemo, useState } from "react";
import { useStore } from "../store";
import { CAT_COLOR, CAT_GLYPH } from "./common";

export default function Library() {
  const specs = useStore((s) => s.specs), cats = useStore((s) => s.categories), addNode = useStore((s) => s.addNode);
  const [q, setQ] = useState(""); const [closed, setClosed] = useState<Record<string, boolean>>({});
  const groups = useMemo(() => {
    const ql = q.trim().toLowerCase(); const out: Record<string, Record<string, any[]>> = {};
    Object.values(specs).forEach((n) => {
      if (ql && !`${n.name} ${n.category} ${n.subcategory} ${n.description} ${n.keywords}`.toLowerCase().includes(ql)) return;
      ((out[n.category] ??= {})[n.subcategory || ""] ??= []).push(n);
    }); return out;
  }, [specs, q]);
  return (
    <aside className="w-[250px] shrink-0 bg-white border-r border-line flex flex-col min-h-0" aria-label="Component library">
      <div className="p-2 border-b border-line"><input className="inp" placeholder="Search components…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search components" />
        <div className="text-[11px] text-mute mt-1">Drag onto the canvas, or double-click to add.</div></div>
      <div className="overflow-y-auto flex-1">
        {cats.filter((c) => groups[c]).map((c) => (
          <section key={c}><button className="w-full flex items-center gap-2 px-2 py-1.5 font-semibold text-left hover:bg-paper" onClick={() => setClosed({ ...closed, [c]: !closed[c] })} aria-expanded={!closed[c]}>
            <span style={{ color: CAT_COLOR[c] }}>{CAT_GLYPH[c]}</span><span className="flex-1">{c}</span><span className="text-mute text-xs">{closed[c] && !q ? "▸" : "▾"}</span></button>
            {(!closed[c] || q) && Object.entries(groups[c]).map(([sub, items]) => (<div key={sub}>
              {sub && <div className="px-3 pt-1 text-[11px] text-mute font-medium">{sub}</div>}
              {items.map((n: any) => (<div key={n.type} draggable onDragStart={(e) => { e.dataTransfer.setData("application/vmls", n.type); e.dataTransfer.effectAllowed = "move"; }}
                onDoubleClick={() => addNode(n.type, { x: 120 + Math.random() * 200, y: 120 + Math.random() * 200 })} title={`${n.description}\nIn: ${n.compatibility.accepts.join(", ") || "—"}  Out: ${n.compatibility.produces.join(", ") || "—"}`}
                className="mx-2 my-0.5 px-2 py-1 rounded border border-transparent hover:border-line hover:bg-paper cursor-grab flex items-center gap-2" tabIndex={0} onKeyDown={(e) => e.key === "Enter" && addNode(n.type, { x: 150, y: 150 })}>
                <span className="w-1.5 h-4 rounded-sm" style={{ background: CAT_COLOR[n.category] }} /><span className="truncate">{n.name}</span></div>))}
            </div>))}
          </section>))}
        {!Object.keys(groups).length && <div className="p-3 text-mute">No components match “{q}”.</div>}
      </div>
    </aside>);
}
