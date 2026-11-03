export type ParamSpec = { name: string; label: string; type: string; default: any; description: string; min?: number | null; max?: number | null; step?: number | null;
  options?: string[] | null; nullable: boolean; advanced: boolean; visible_if?: Record<string, any[]> | null };
export type PortSpec = { name: string; type: string; label: string; multi: boolean; required: boolean };
export type NodeSpec = { type: string; name: string; category: string; subcategory: string; description: string; icon: string; inputs: PortSpec[]; outputs: PortSpec[];
  params: ParamSpec[]; doc: Record<string, string>; example: string; keywords: string; compatibility: { accepts: string[]; produces: string[] } };
export type NodeState = { status: string; duration: number | null; logs: string[]; warnings: string[]; error: { message: string; technical: string | null } | null; preview: any; cached: boolean };
export type Job = { id: string; status: string; phase: string; progress: number; logs: { node_id: string | null; text: string }[]; issues: any[]; nodes: Record<string, NodeState>; started: number; finished: number | null };
export type Issue = { level: string; node_id?: string; edge_id?: string; message: string };
