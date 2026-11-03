# Visual ML Studio Implementation Progress

## Completed and verified

- [x] Backend startup and health endpoint
- [x] Frontend startup and production build
- [x] Drag/drop node editor
- [x] Typed connection validation and cycle rejection
- [x] Edge disconnect and endpoint rewiring
- [x] Undo/redo for graph changes
- [x] Workflow validation and topological execution
- [x] Node-level status, logs, previews, warnings, and errors
- [x] Branching workflows and model comparison
- [x] CSV, TSV, Excel, Parquet, JSON, and JSONL upload paths
- [x] Quoted semicolon CSV delimiter detection
- [x] Repair of legacy uploads stored with one combined column
- [x] Scrollable checkbox feature selection
- [x] Target-column list, suggested target action, and clear target action
- [x] Duplicate-row removal and downstream rewiring
- [x] Classical preprocessing and model nodes exposed by the registry
- [x] Metrics and Plotly chart generation
- [x] Python and Jupyter notebook export for supported workflows
- [x] Project save/restore and dataset references
- [x] Workflow/model configuration persistence
- [x] Backend regression tests (49 passing)
- [x] Frontend TypeScript/Vite build
- [x] Frontend integration checks
- [x] Anomaly detection (Isolation Forest, LOF, One-Class SVM) with executable code export
- [x] t-SNE and Truncated SVD dimensionality reduction with executable code export
- [x] Optional PyTorch tabular neural networks: MLP, CNN, LSTM, GRU, and Transformer
- [x] PyTorch execution, evaluation, and generated-script tests

## Not implemented in the current architecture

- [ ] Optional UMAP node (requires the optional `umap-learn` dependency)
- [ ] Independent save/load of fitted model artifacts
- [ ] Checked-in browser end-to-end suite

## Verification commands

```powershell
cd backend
python -m pytest

cd ..\frontend
npm run build
npm run test:integration
```

## Honest release status

The classical visual ML laboratory is functional and tested. The attached requirements also request a deep-learning laboratory and several additional algorithm families that are not present in the existing codebase; those remain explicitly tracked above rather than being represented as fake or incomplete UI nodes.
