# Architecture

**Graph model.** A workflow is `{nodes:[{id,type,params}], edges:[{source,sourceHandle,target,targetHandle}]}` (Pydantic, `workflow/graph.py`). Handles are port names. The server re-validates everything (`workflow/validator.py`): unknown types, parameter ranges/options, port-type compatibility, single-input ports, required inputs, cycles.

**Port types.** `dataset` → `split` → `trained` → `metrics` / `figure`. A *split* carries raw train/test frames plus an ordered list of **preprocessing step recipes**; steps are turned into sklearn objects only inside `Pipeline`s (`nodes/steps.py`), so preprocessing is always fitted on training rows (and inside every CV fold). The same step plan generates both the sklearn object and the Python source, so exported code cannot drift from what ran.

**Execution.** `workflow/executor.py`: validate → ancestors-only plan for partial runs → topological order → per node: cache lookup (`sha256(type, params, upstream keys)`) → `spec.run` → preview JSON. Changing a node changes its key and all downstream keys, so only dependents re-run. Failure of a node marks downstream nodes *skipped* and keeps earlier results. Jobs run in a thread pool; the UI polls `/api/jobs/{id}`. Cancellation is cooperative between nodes.

**Registry.** Each `NodeSpec` has: type id, name, category, ports, `Param` list (drives the inspector form and server validation), `run`, `codegen`, educational `doc`. 28 model nodes are generated from a table in `nodes/models.py`.

**Code generation.** `codegen/generator.py` walks the same topological order and calls each node's `codegen`, collecting imports and (title, markdown, code) cells → `.py` or `.ipynb` (nbformat + `validate_notebook`: schema, syntax of every cell, required imports). Built-in datasets are embedded as the same code snippet the backend executes.

**Frontend.** Zustand store holds graph, selection, undo/redo snapshots, job state. React Flow renders nodes; `isValid` mirrors server rules for instant feedback. Execution state lives in `job`, separate from graph state.
