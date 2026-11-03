# Adding a node

1. Create a function `run(ctx, params, inputs) -> NodeResult(outputs, preview, warnings)`; raise `NodeError(friendly_message, technical)` for user mistakes; use `ctx.log()` for progress and `ctx.check_cancel()` in long loops.
2. Create `codegen(c, node_id, params, ins) -> {port: var}`; call `c.cell(node_id, title, markdown, code)` and add imports to `c.imports`. Variables are named from `c.out(node_id, port)`; propagate `c.meta[var]` (target/task/seed) for downstream nodes.
3. `register(NodeSpec(type, name, category, description, inputs, outputs, [Param(...)], run, codegen, doc={...}))`.
4. Import the module in `nodes/__init__.py`. The node automatically appears in the library, inspector, validator and exports.
5. Add a test: run a small workflow with `tests/helpers.run` and, if it generates code, execute `to_script` and compare against the app's metrics.

Params use `type` in `int|float|bool|select|string|text|column|columns`; `visible_if={"other": [values]}` shows a control conditionally; `map` converts UI values to sklearn values (used for both execution and code).
