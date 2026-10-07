import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# отдельное хранилище дашбордов для тестов
_tmp = tempfile.mkdtemp(prefix="bibot-tests-")
os.environ["APP_DB"] = str(Path(_tmp) / "app.db")
os.environ.setdefault("NLU_MODE", "rules")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def catalog():
    from app.semantic import catalog as cat

    return cat()
