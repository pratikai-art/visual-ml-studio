# Visual ML Studio

Visual ML Studio is a browser-based, drag-and-drop machine-learning laboratory. It lets you build workflows from connected nodes, run real pandas and scikit-learn code, inspect data and metrics, compare experiments, and export runnable Python or Jupyter notebooks.

The application has:

- A React + TypeScript frontend using React Flow, Zustand, Tailwind CSS, and Plotly.
- A FastAPI backend that validates and executes workflows.
- Built-in educational datasets and support for uploaded CSV, TSV, Excel, Parquet, JSON, and JSON Lines files.
- A workflow engine with validation, topological execution, caching, node-level logs, progress, cancellation, and friendly errors.
- Project export/import and experiment history.

No model result is hard-coded or simulated. The backend executes the selected workflow using real data and machine-learning libraries.

## Requirements

### Required

- Python 3.10 or newer
- Node.js 18 or newer
- npm

### Supported operating systems

The project is designed to run on Windows, macOS, and Linux. The commands below use the Windows PowerShell style where platform-specific commands are needed.

## Project structure

```text
visual-ml-studio/
├── backend/
│   ├── app/
│   │   ├── api/             FastAPI routes
│   │   ├── codegen/         Python and notebook generation
│   │   ├── datasets/        Upload storage and built-in datasets
│   │   ├── experiments/     Run history
│   │   ├── nodes/           Dataset, cleaning, preprocessing, model, and chart nodes
│   │   ├── utils/           JSON and dataframe helpers
│   │   └── workflow/        Specs, validation, graph planning, cache, and executor
│   ├── tests/
│   │   ├── integration/
│   │   ├── notebook_export/
│   │   └── unit/
│   ├── data/                Runtime uploads, projects, and history
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── components/      Editor, inspector, library, top bar, and results panel
│   │   ├── api.ts           Backend API client
│   │   ├── store.ts         Zustand workflow state
│   │   ├── templates.ts     Example workflows
│   │   └── App.tsx
│   ├── tests/
│   ├── package.json
│   └── vite.config.ts
├── docs/
├── examples/
├── Dockerfile
└── docker-compose.yml
```

## Quick start on Windows

Open two PowerShell terminals.

### 1. Install and start the backend

```powershell
cd C:\Users\<you>\Downloads\visual-ml-studio\visual-ml-studio\backend

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[test]"

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install and start the frontend

```powershell
cd C:\Users\<you>\Downloads\visual-ml-studio\visual-ml-studio\frontend
npm install
npm run dev
```

Open the URL printed by Vite, normally:

```text
http://127.0.0.1:5173
```

The Vite development server proxies `/api` requests to `http://127.0.0.1:8000`.

### 3. Check the backend

Open this URL in a browser:

```text
http://127.0.0.1:8000/api/health
```

A healthy response includes the backend status, pandas version, scikit-learn version, and upload limit.

## Run the built application with the backend only

The backend serves the compiled frontend from `frontend/dist` when that directory exists.

```powershell
cd C:\Users\<you>\Downloads\visual-ml-studio\visual-ml-studio\frontend
npm install
npm run build

cd ..\backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

Use the Vite development server when changing frontend source files. Rebuild `frontend/dist` before using the backend-only mode.

## Docker

From the project root:

```powershell
docker compose up --build
```

Open:

```text
http://localhost:8000
```

The Docker setup:

- Builds the frontend with Node.
- Installs the backend in a Python image.
- Serves the compiled UI and API from port 8000.
- Stores runtime data in the `vmls-data` Docker volume.

Stop the containers with:

```powershell
docker compose down
```

## First workflow

1. Start the backend and frontend.
2. Open Visual ML Studio.
3. The Student Scores regression template is loaded by default.
4. Review the graph on the canvas.
5. Click **Run Workflow**.
6. Watch the node statuses and execution logs.
7. Select a node and inspect:
   - Data Preview
   - Metrics
   - Charts
   - Predictions
   - Errors & Warnings
   - Generated Code
8. Use **Save** to download a `.vmls.json` project file.

### Connecting nodes

- Drag an output handle on the right side of a node to a compatible input handle on the left side of another node.
- Port colors identify data types.
- Invalid type connections and cycles are rejected.
- Drag either end of an existing edge to rewire it.
- Drag an edge endpoint into empty space to disconnect it.
- Select an edge and press `Delete` or `Backspace` to remove it.
- Use `Ctrl+Z` and `Ctrl+Y` for undo and redo.

## Working with datasets

### Built-in datasets

The application includes educational datasets such as classification, regression, clustering, and dimensionality-reduction examples. Add a Dataset Input node and select a built-in dataset in the Inspector.

### Uploading a dataset

Click **Upload Dataset** and select or drop a file. Supported formats:

- CSV
- TSV
- Excel `.xlsx`
- Parquet
- JSON
- JSON Lines `.jsonl`

Uploaded files are stored locally under `backend/data/uploads`. They are not sent to an external service.

CSV delimiter detection supports comma, semicolon, tab, and pipe-separated files, including quoted headers. The detected delimiter is retained for workflow execution and generated code.

### Selecting features and a target

Use the **Select Features & Target** node:

1. Choose the target column from the scrollable target list.
2. Choose classification, regression, or automatic task detection.
3. Tick the feature columns to include.
4. Use **Select all**, **Deselect all**, **Numeric**, or **Non-numeric** for quick selection.
5. Click **Apply configuration**.

If no feature list is selected, the node uses every column except the target. The target is always excluded from the feature matrix.

### Duplicate rows

To remove duplicates:

1. Upload or select the dataset.
2. Add a **Duplicate Rows** node, or use **Remove duplicate rows in workflow** after upload.
3. Set **Mode** to `remove`.
4. Choose whether to keep the first row, last row, or remove all copies.
5. Connect the cleaned output to the rest of the workflow.

Duplicate detection in Dataset Explorer describes the data at that point in the graph. To make later nodes use cleaned data, the Duplicate Rows output must be connected before those nodes.

## Recommended supervised-learning workflow

```text
Dataset Input
  → Duplicate Rows (optional)
  → Dataset Explorer (optional)
  → Select Features & Target
  → Train/Test Split
  → Imputer / Encoder / Scaler (as needed)
  → Model
  → Model Evaluation
  → Model Chart (optional)
```

Preprocessing is applied after the split and is fitted on training data through scikit-learn pipelines to reduce data leakage.

## Available workflow areas

### Data

- Dataset Input
- Dataset Explorer
- Missing-Value Analysis

### Cleaning

- Duplicate Rows
- Drop Columns
- Drop Missing
- Outlier handling
- Constant-column removal

### Feature selection and preprocessing

- Select Features & Target
- SelectKBest
- Variance threshold
- Model-based selection
- RFE
- Imputer
- Scaler and transformer
- One-hot and ordinal encoding

### Splitting

- Hold-out train/test split
- Stratified split
- Chronological split
- Cross-validation options on model nodes

### Models

The project includes multiple real scikit-learn classifiers and regressors, including logistic regression, SVC, KNN, random forest, ridge, linear regression, gradient boosting, and others.

### Unsupervised learning

- K-Means
- MiniBatch K-Means
- DBSCAN
- Agglomerative clustering
- Gaussian mixture models
- PCA

### Evaluation and visualization

- Classification metrics
- Regression metrics
- Confusion matrix
- ROC and precision-recall curves
- Residuals
- Feature importance
- Prediction and data charts
- Model comparison

## Templates

The Templates menu provides ready-to-edit workflows for:

- Student-score regression
- Classification
- Regression and classification benchmarks
- Scaling comparisons
- Missing-value strategies
- Feature-selection comparisons
- Cross-validation
- Hyperparameter optimization
- Clustering
- PCA visualization

Templates are ordinary editable graphs. You can change parameters, delete nodes, add nodes, and rewire connections.

## Export and project files

### Generated code

The **Generated Code** tab creates runnable Python based on the workflow. It includes the selected data-loading, preprocessing, model, evaluation, and chart steps.

### Notebook export

Download a Jupyter notebook when you want to run the experiment outside the application. Uploaded datasets are referenced by filename and are intentionally not embedded in exported code.

Place the dataset beside the generated notebook or update the generated `DATASET_PATH`.

### Project files

Save creates a versioned `.vmls.json` project containing:

- Workflow name
- Node types and positions
- Node parameters
- Connections
- Mode settings
- Journal entries
- References to uploaded datasets
- Selected result summaries

Project files reference datasets by content hash. If a referenced upload is missing, re-upload the same file or reconnect the Dataset Input node.

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+C` | Copy selected nodes |
| `Ctrl+V` | Paste nodes |
| `Ctrl+D` | Duplicate selected nodes |
| `Delete` / `Backspace` | Delete selected nodes or edges |
| `Ctrl+Enter` | Run workflow |
| `Ctrl+S` | Save project |
| `Shift` + drag | Box-select nodes |

## Tests and validation

### Backend tests

```powershell
cd backend
pytest -q
```

The backend suite covers workflow execution, validation, cleaning, preprocessing, models, metrics, code generation, notebook export, and project behavior.

### Frontend build

```powershell
cd frontend
npm run build
```

This runs TypeScript checking followed by the Vite production build.

### Frontend integration checks

Start the backend first, then run:

```powershell
cd frontend
npm run test:integration
```

These checks exercise templates, connection validation, parameter updates, undo/redo, project round trips, and layout behavior.

## Configuration

The backend supports these environment variables:

| Variable | Default | Description |
|---|---:|---|
| `VMLS_DATA_DIR` | `backend/data` | Runtime uploads, projects, and history directory |
| `VMLS_MAX_UPLOAD_MB` | `100` | Maximum upload size |
| `VMLS_JOB_TIMEOUT_S` | `900` | Maximum execution time for a workflow |

Example:

```powershell
$env:VMLS_DATA_DIR = "D:\visual-ml-data"
$env:VMLS_MAX_UPLOAD_MB = "250"
$env:VMLS_JOB_TIMEOUT_S = "1800"
python -m uvicorn app.main:app --port 8000
```

## Troubleshooting

### Blank page or “could not render”

1. Make sure the backend is running on port 8000.
2. Make sure the frontend is running on port 5173 when using development mode.
3. Refresh the browser after rebuilding.
4. Check the browser console for the exact error.
5. Run `npm run build` to catch TypeScript errors.

### Port already in use

Check which process owns a port:

```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen
Get-NetTCPConnection -LocalPort 5173 -State Listen
```

If the application is already responding, use the existing server instead of starting another copy.

### Feature list shows one combined column

This usually means the CSV delimiter was not detected correctly or the browser is using an old backend process.

1. Restart the backend.
2. Refresh the browser.
3. Open the dataset profile again.
4. Confirm the columns are shown separately.
5. Re-upload the file only if the existing upload cannot be repaired.

Semicolon-separated files should show separate columns such as `fixed acidity`, `alcohol`, and `quality`.

### “No target column chosen”

Select a target in the Select Features & Target node, then click **Apply configuration** before running the workflow. For the wine-quality example, `quality` is typically the target.

### Duplicate warning remains

Dataset Explorer reports duplicates in the dataset that reaches it. Ensure the Duplicate Rows node is connected upstream of Dataset Explorer and all model branches. Run the workflow again after applying the configuration.

### Backend API cannot be reached

Check:

```text
http://127.0.0.1:8000/api/health
```

If it fails, start the backend from the `backend` directory:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Current limitations

- Jobs run in worker threads inside the API process.
- Stop requests take effect at node boundaries; a model currently fitting may finish first.
- Separate worker processes and strict memory limits are not implemented.
- PyTorch tabular neural-network workflows are available through the optional `pytorch_neural_network` node. Install them with `pip install -e ".[deep-learning]"`; the node supports MLP, CNN, LSTM, GRU, and Transformer architectures.
- Image tensors, raw text tensors, GPU orchestration, and arbitrary PyTorch tensor graphs are outside the current dataset contract.
- Some optional libraries are listed separately and are not required for the core application.
- The frontend uses polling for live job updates rather than WebSockets or server-sent events.

## Data and privacy

Uploaded datasets are stored in the local backend data directory. The application does not submit user data to a third-party service. Treat the runtime data directory as sensitive if datasets contain private or personal information.

## License and dataset attribution

Built-in dataset source and license information is displayed in the application. The sample student-score data is synthetic. Review the source and license information before redistributing generated notebooks or datasets.

## Additional documentation

- [User guide](docs/user_guide.md)
- [Architecture](docs/architecture.md)
- [Node development](docs/node_development.md)
- [Student scores example](examples/student_scores/)
- [Implementation audit](IMPLEMENTATION_AUDIT.md)
- [Implementation progress](IMPLEMENTATION_PROGRESS.md)
