# Visual ML Studio Implementation Audit

## Scope

This audit compares the attached completion requirements with the current source tree, backend tests, frontend integration checks, and the verified browser workflow.

## Status summary

| Area | Status | Evidence |
|---|---|---|
| Application startup and API health | Complete | `backend/app/main.py`, `/api/health`, Vite proxy |
| Visual graph editing | Complete | React Flow editor, node library, inspector, minimap, undo/redo |
| Connect, disconnect, and rewire | Complete | Controlled edges, React Flow reconnect handlers, cycle/type checks |
| Workflow validation and execution | Complete | `workflow/validator.py`, `workflow/executor.py`, integration suite |
| Dataset upload and profiling | Complete | CSV/TSV/XLSX/Parquet/JSON/JSONL support and profile endpoint |
| Quoted semicolon CSV compatibility | Fixed | delimiter detection and legacy-upload repair in `datasets/store.py` |
| Feature and target selection | Complete | scrollable target/feature selectors with checkbox controls |
| Data cleaning and preprocessing | Complete for exposed nodes | cleaning nodes and sklearn pipeline steps are covered by tests |
| Classical machine learning | Complete for exposed nodes | registry-driven sklearn model nodes and workflow tests |
| Evaluation and visualization | Complete for exposed nodes | metrics and Plotly figures are generated from execution outputs |
| Model comparison and history | Complete | comparison node and experiment history routes/UI |
| Python and notebook export | Complete for supported workflows | codegen tests and fresh-kernel notebook test |
| Project save/restore | Complete | project routes, content-hash dataset references, tests |
| Trained-model artifact persistence | Not implemented | project persistence stores workflow configuration, not fitted estimator files |
| Deep learning/PyTorch | Complete for tabular workflows | optional PyTorch dependency, real training node, evaluation integration, and export tests |
| CNN/LSTM/GRU/Transformer | Complete for tabular workflows | one registry node exposes all four architectures and each has execution coverage |
| Anomaly detection | Complete | `anomaly_detection` node supports Isolation Forest, LOF, and One-Class SVM with execution and codegen tests |
| t-SNE/Truncated SVD/UMAP | Partial | t-SNE and Truncated SVD are implemented and tested; optional UMAP remains unavailable |
| Browser end-to-end automation | Partial | browser verification performed manually; no checked-in Playwright suite |

## Important bugs found and repaired

1. Quoted semicolon-separated CSV files were parsed as one column. The parser now counts delimiters in sample lines, repairs legacy metadata, and preserves the delimiter for generated code.
2. The feature selector could only show the combined header when an old upload had been misparsed. Existing uploads are reparsed on load.
3. Target selection now supports a suggested target action and a scrollable list.
4. Duplicate-removal workflow insertion now rewires all downstream branches through the cleaning node.
5. Explorer no longer reports the raw duplicate warning as a downstream cleaning failure.
6. React effect cleanup previously returned a non-function value; this caused `destroy is not a function` and a blank screen. The effect now has an explicit body.

## Testing gaps

- Image/text tensor workflows cannot be tested because the current dataset contract is tabular.
- Browser interactions are not currently represented by a checked-in automated browser test.
- Fitted model artifacts are not saved and loaded independently of an execution job.
- Optional libraries listed in `pyproject.toml` are not part of the core test matrix.
- The frontend integration script exercises store behavior and a running backend, but not every rendered component.

## Prioritized remaining work

1. Add checked-in browser automation for upload, feature selection, connect/rewire, run, and export.
2. Add optional UMAP only when the optional dependency is installed and supported by the release.
3. Add richer transformed-data filtering/sorting controls if required by the product specification.

## Architecture notes

The current project uses a registry-driven node architecture:

- `workflow/spec.py` defines typed ports, parameters, node results, and registration.
- `workflow/validator.py` validates graph structure and parameters.
- `workflow/executor.py` performs topological execution and maintains typed in-memory artifacts.
- `nodes/` contains domain operations and code-generation functions.
- `api/routes.py` is the HTTP infrastructure layer.
- `frontend/src/store.ts` serializes the graph and coordinates UI state.

The existing implementation intentionally uses pandas and scikit-learn artifacts inside the worker process and does not serialize fitted model objects through JSON.
