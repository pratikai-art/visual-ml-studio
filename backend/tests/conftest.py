import os, sys, tempfile
from pathlib import Path
os.environ.setdefault("VMLS_DATA_DIR", tempfile.mkdtemp(prefix="vmls_test_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from app.workflow.executor import CACHE


@pytest.fixture(autouse=True)
def _clear():
    CACHE.clear()
    yield
