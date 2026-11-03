const j = async (r: Response) => {
  if (!r.ok) { let m = r.statusText; try { const d = await r.json(); m = typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail); } catch {} throw new Error(m); }
  return r.json();
};
const post = (u: string, b: any) => fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) });
export const api = {
  nodes: () => fetch("/api/nodes").then(j),
  builtin: () => fetch("/api/builtin-datasets").then(j),
  datasets: () => fetch("/api/datasets").then(j),
  upload: (f: File) => { const fd = new FormData(); fd.append("file", f); return fetch("/api/datasets", { method: "POST", body: fd }).then(j); },
  profile: (id: string) => fetch(`/api/datasets/${id}`).then(j),
  validate: (wf: any) => post("/api/workflow/validate", wf).then(j),
  run: (workflow: any, target_node?: string) => post("/api/workflow/run", { workflow, target_node: target_node ?? null }).then(j),
  job: (id: string) => fetch(`/api/jobs/${id}`).then(j),
  cancel: (id: string) => post(`/api/jobs/${id}/cancel`, {}).then(j),
  columns: (workflow: any, node_id: string) => post("/api/workflow/columns", { workflow, node_id }).then(j),
  code: (wf: any, node?: string) => post(`/api/workflow/code${node ? `?node_id=${node}` : ""}`, wf).then(j),
  download: async (kind: "notebook" | "script", wf: any, name: string) => {
    const r = await post(`/api/workflow/${kind}`, wf);
    if (!r.ok) throw new Error((await r.json()).detail);
    const blob = await r.blob(); const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = `${name.replace(/[^A-Za-z0-9_-]+/g, "_") || "experiment"}.${kind === "notebook" ? "ipynb" : "py"}`; a.click(); URL.revokeObjectURL(a.href);
  },
  saveProject: (p: any) => post("/api/projects", p).then(j),
  projects: () => fetch("/api/projects").then(j),
  loadProject: (slug: string) => fetch(`/api/projects/${slug}`).then(j),
  checkProject: (p: any) => post("/api/projects/check", p).then(j),
  history: () => fetch("/api/history").then(j),
  clearHistory: () => fetch("/api/history", { method: "DELETE" }).then(j),
};
