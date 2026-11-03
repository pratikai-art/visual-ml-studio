import { Handle, Position, type NodeProps } from "@xyflow/react";
import { useStore, type RFNode } from "../store";
import { CAT_COLOR, CAT_GLYPH, PORT_COLOR, STATUS, fmt } from "./common";

export default function NodeCard({ id, data, selected }: NodeProps<RFNode>) {
  const spec = useStore((s) => s.specs[data.type]); const st = useStore((s) => s.job?.nodes[id]);
  const issue = useStore((s) => s.issues.find((i) => i.node_id === id && i.level === "error"));
  if (!spec) return <div className="bg-red-100 p-2 rounded">Unknown node {data.type}</div>;
  const color = CAT_COLOR[spec.category] || "#444"; const status = st?.status ?? "not_run"; const S = STATUS[status] ?? STATUS.not_run;
  const changed = spec.params.filter((p) => JSON.stringify(data.params[p.name]) !== JSON.stringify(p.default) && p.type !== "text").slice(0, 3);
  const pv = st?.preview; const rows = Math.max(spec.inputs.length, spec.outputs.length, 1);
  return (
    <div className="bg-white rounded-lg shadow-sm w-[236px]" style={{ border: `1.5px solid ${selected ? "#0f766e" : status === "failed" ? "#b91c1c" : "#cfd3cb"}`, boxShadow: selected ? "0 0 0 3px rgba(15,118,110,.18)" : undefined }}>
      <div className="flex items-center gap-2 px-2 py-1.5 rounded-t-lg text-white" style={{ background: color }}>
        <span aria-hidden className="text-base leading-none">{CAT_GLYPH[spec.category]}</span><span className="font-semibold truncate flex-1">{spec.name}</span>
        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-white/90 font-medium" style={{ color: S.color }}>{status === "running" ? "● running" : S.label}</span></div>
      <div className="relative" style={{ minHeight: rows * 22 + 6 }}>
        {spec.inputs.map((p, i) => (<div key={p.name}><Handle type="target" position={Position.Left} id={p.name} style={{ top: 18 + i * 22, background: PORT_COLOR[p.type] }} />
          <div className="absolute left-3 text-[10px] text-mute" style={{ top: 11 + i * 22 }}>{p.label}{p.multi ? " ×n" : ""}</div></div>))}
        {spec.outputs.map((p, i) => (<div key={p.name}><Handle type="source" position={Position.Right} id={p.name} style={{ top: 18 + i * 22, background: PORT_COLOR[p.type] }} />
          <div className="absolute right-3 text-[10px] text-mute text-right" style={{ top: 11 + i * 22 }}>{p.label}</div></div>))}
      </div>
      <div className="px-2 pb-2 text-[11px] space-y-0.5">
        {changed.length > 0 && <div className="text-mute truncate">{changed.map((p) => `${p.label}: ${Array.isArray(data.params[p.name]) ? data.params[p.name].length + " cols" : String(data.params[p.name])}`).join(" · ")}</div>}
        {st?.duration != null && <div className="text-mute">{st.cached ? "cached" : `${st.duration.toFixed(2)}s`}</div>}
        {pv?.kind === "metrics" && <div className="mono">{(pv.task === "regression" ? ["r2", "rmse"] : ["accuracy", "f1"]).map((k) => `${k} ${fmt(pv.metrics[k])}`).join("  ")}</div>}
        {pv?.shape && <div className="mono text-mute">{pv.shape[0]}×{pv.shape[1]}</div>}
        {pv?.kind === "split" && <div className="mono text-mute">{pv.n_train} train / {pv.n_test} test</div>}
        {(st?.warnings?.length ?? 0) > 0 && <div className="text-amber-700 truncate" title={st!.warnings.join("\n")}>⚠ {st!.warnings[0]}</div>}
        {st?.error && <div className="text-red-700 line-clamp-3" title={st.error.message}>✕ {st.error.message}</div>}
        {!st?.error && issue && <div className="text-red-700 line-clamp-2">✕ {issue.message}</div>}
      </div>
    </div>);
}
