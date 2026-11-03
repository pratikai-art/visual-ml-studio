const base = "http://127.0.0.1:8000";
const of = globalThis.fetch; (globalThis as any).fetch = (u: any, o?: any) => of(typeof u === "string" && u.startsWith("/") ? base + u : u, o);
import { useStore } from "../src/store";
import { TEMPLATES } from "../src/templates";
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
(async () => {
  const s = () => useStore.getState(); await s().init(); let fails = 0;
  for (let i = 0; i < TEMPLATES.length; i++) {
    s().loadTemplate(i);
    const wf = s().workflow();
    const bad = s().edges.filter((e) => !s().isValid({ ...e } as any) && true);
    await s().run(); let j; for (let k = 0; k < 200; k++) { await sleep(250); j = s().job; if (j && (j.status === "Completed" || j.status === "Failed")) break; }
    const errs = Object.entries(j!.nodes).filter(([, v]: any) => v.error).map(([k, v]: any) => k + ": " + v.error.message);
    const ok = j!.status === "Completed"; if (!ok) fails++;
    console.log(ok ? "OK  " : "FAIL", TEMPLATES[i].name, `${s().nodes.length} nodes`, errs.join(" | "));
  }
  // store behaviours
  s().loadTemplate(1); const n0 = s().nodes.length;
  console.log("invalid conn (dataset→model split):", s().isValid({ source: "d", target: "m", sourceHandle: "dataset", targetHandle: "split" } as any));
  console.log("cycle conn rejected:", !s().isValid({ source: "ev", target: "d", sourceHandle: "metrics", targetHandle: "dataset" } as any));
  s().select("m"); s().setParams("m", { C: 5 }); console.log("param applied:", s().nodes.find((n) => n.id === "m")!.data.params.C);
  s().undo(); console.log("undo restores C:", s().nodes.find((n) => n.id === "m")!.data.params.C); s().redo(); console.log("redo:", s().nodes.find((n) => n.id === "m")!.data.params.C);
  s().addNode("clf_knn", { x: 0, y: 0 }); console.log("added:", s().nodes.length === n0 + 1);
  const p = s().exportProject(); await s().importProject(JSON.parse(JSON.stringify(p))); console.log("project roundtrip nodes:", s().nodes.length === n0 + 1, "format", p.format, p.version);
  console.log("autoLayout:"); s().autoLayout(); console.log(s().nodes.slice(0, 3).map((n) => n.position));
  process.exit(fails ? 1 : 0);
})();
