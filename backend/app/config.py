import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("VMLS_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
PROJECT_DIR = DATA_DIR / "projects"
HISTORY_FILE = DATA_DIR / "history.json"
MAX_UPLOAD_MB = int(os.environ.get("VMLS_MAX_UPLOAD_MB", "100"))
JOB_TIMEOUT_S = int(os.environ.get("VMLS_JOB_TIMEOUT_S", "900"))
ALLOWED_EXT = {".csv", ".tsv", ".xlsx", ".parquet", ".json", ".jsonl"}
for d in (UPLOAD_DIR, PROJECT_DIR):
    d.mkdir(parents=True, exist_ok=True)
