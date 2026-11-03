import { useCallback, useEffect, useState } from "react";
import { ReactFlow, ReactFlowProvider, Background, Controls, MiniMap, useReactFlow, SelectionMode } from "@xyflow/react";
import { useStore } from "./store";
import NodeCard from "./components/NodeCard";
import Library from "./components/Library";
import Inspector from "./components/Inspector";
import BottomPanel from "./components/BottomPanel";
import Topbar from "./components/Topbar";
import { CAT_COLOR } from "./components/common";

const nodeTypes = { card: NodeCard };

function Canvas() {
  const s = useStore(); const rf = useReactFlow(); const [menu, setMenu] = useState<{ x: number; y: number; id: string } | null>(null);
  const onDrop = useCallback((e: React.DragEvent) => { e.preventDefault(); const t = e.dataTransfer.getData("application/vmls"); if (t) s.addNode(t, rf.screenToFlowPosition({ x: e.clientX - 100, y: e.clientY - 20 })); }, [rf]);
  const edges = s.edges.map((e) => { const st = s.job?.nodes[e.source]?.status; return { ...e, reconnectable: true, animated: st === "running" || s.job?.nodes[e.target]?.status === "running", style: { stroke: st === "failed" ? "#b91c1c" : "#6b7a86", strokeWidth: 1.6 } }; });
  return (<div className="flex-1 relative" onDrop={onDrop} onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; }}>
    <ReactFlow nodes={s.nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={s.onNodesChange} onEdgesChange={s.onEdgesChange} onConnect={s.onConnect} onReconnect={s.onReconnect} edgesReconnectable
      onReconnectEnd={(_, edge, __, state) => { if (!state.isValid && !state.toNode) s.disconnect(edge.id); }} isValidConnection={s.isValid as any}
      onNodeClick={(_, n) => { s.select(n.id); setMenu(null); }} onPaneClick={() => { s.select(null); setMenu(null); }} onNodeContextMenu={(e, n) => { e.preventDefault(); s.select(n.id); setMenu({ x: e.clientX, y: e.clientY, id: n.id }); }}
      fitView minZoom={0.15} maxZoom={2} deleteKeyCode={["Delete", "Backspace"]} selectionOnDrag={false} selectionKeyCode="Shift" multiSelectionKeyCode={["Control", "Meta"]} selectionMode={SelectionMode.Partial} proOptions={{ hideAttribution: true }}>
      <Background gap={20} color="#dcdfd8" /><Controls /><MiniMap pannable zoomable nodeColor={(n: any) => CAT_COLOR[s.specs[n.data?.type]?.category] || "#999"} />
    </ReactFlow>
    {!s.nodes.length && <div className="absolute inset-0 flex items-center justify-center pointer-events-none text-mute text-center"><div><div className="text-lg font-semibold text-ink">Empty canvas</div>Drag a component from the library, or pick a template in the top bar.</div></div>}
    {menu && <div className="fixed z-50 bg-white border border-line rounded shadow-lg py-1 w-44" style={{ left: menu.x, top: menu.y }} onMouseLeave={() => setMenu(null)}>
      {[["Run up to this node", () => s.run(menu.id)], ["Duplicate", () => { s.duplicateSelected(); }], ["Copy", s.copy], ["Delete", s.deleteSelected]].map(([l, f]: any) => <button key={l} className="block w-full text-left px-3 py-1 hover:bg-paper" onClick={() => { f(); setMenu(null); }}>{l}</button>)}</div>}
    <div className="absolute top-2 left-2 flex gap-1 z-10">{[["Undo", s.undo, !s.past.length], ["Redo", s.redo, !s.future.length], ["Auto layout", s.autoLayout, !s.nodes.length]].map(([l, f, d]: any) => <button key={l} className="btn shadow-sm" disabled={d} onClick={f}>{l}</button>)}</div>
  </div>);
}

export default function App() {
  const ready = useStore((s) => s.ready); const init = useStore((s) => s.init); const toasts = useStore((s) => s.toasts); const [err, setErr] = useState("");
  useEffect(() => { init().catch((e) => setErr(String(e.message || e))); }, []);
  useEffect(() => {
    const k = (e: KeyboardEvent) => { const st = useStore.getState(); const tag = (e.target as HTMLElement)?.tagName; if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return; const c = e.ctrlKey || e.metaKey;
      if (c && e.key === "z") { e.preventDefault(); st.undo(); } else if (c && (e.key === "y" || (e.shiftKey && e.key === "Z"))) { e.preventDefault(); st.redo(); } else if (c && e.key === "c") st.copy(); else if (c && e.key === "v") st.paste();
      else if (c && e.key === "d") { e.preventDefault(); st.duplicateSelected(); } else if (c && e.key === "Enter") { e.preventDefault(); st.run(); } else if (c && e.key === "s") { e.preventDefault(); window.dispatchEvent(new Event("vmls-save")); } };
    window.addEventListener("keydown", k); return () => window.removeEventListener("keydown", k);
  }, []);
  if (err) return <div className="p-8"><h1 className="text-lg font-bold">Cannot reach the backend</h1><p className="mt-2">{err}</p><p className="text-mute mt-2">Start it with <code className="mono">uvicorn app.main:app --port 8000</code> from the backend folder.</p></div>;
  if (!ready) return <div className="p-8 text-mute">Loading components…</div>;
  return (<ReactFlowProvider><div className="h-full flex flex-col"><Topbar />
    <div className="flex flex-1 min-h-0"><Library /><div className="flex-1 flex flex-col min-w-0"><Canvas /><BottomPanel /></div><Inspector /></div>
    <div className="fixed bottom-3 right-3 space-y-2 z-[60]" aria-live="polite">{toasts.map((t) => <div key={t.id} className={`px-3 py-2 rounded shadow-lg text-white max-w-sm ${t.kind === "error" ? "bg-red-700" : t.kind === "ok" ? "bg-green-700" : "bg-slate-800"}`}>{t.text}</div>)}</div></div></ReactFlowProvider>);
}
