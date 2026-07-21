import os
import sys

# Make sure `import app` finds backend/app.py regardless of the cwd pytest
# was invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture
def client(monkeypatch):
    """Flask test client with training state reset between tests and no
    API key set, so the un-authed tests exercise the "open" default."""
    monkeypatch.delenv("API_KEY", raising=False)
    import app as flask_app_module
    flask_app_module.API_KEY = ""
    flask_app_module.app.config.update(TESTING=True)
    _reset_state(flask_app_module)
    with flask_app_module.app.test_client() as c:
        yield c


@pytest.fixture
def authed_client(monkeypatch):
    """Same app, but with an API key configured — used to test that the
    training endpoints are actually gated."""
    import app as flask_app_module
    flask_app_module.API_KEY = "test-secret-key"
    flask_app_module.app.config.update(TESTING=True)
    _reset_state(flask_app_module)
    with flask_app_module.app.test_client() as c:
        yield c
    flask_app_module.API_KEY = ""


def _reset_state(flask_app_module):
    flask_app_module.state["image"] = {
        "running": False, "epoch": 0, "total": 0, "d_loss": [], "g_loss": [], "status": "idle"
    }
    flask_app_module.state["text"] = {
        "running": False, "epoch": 0, "total": 0, "d_loss": [], "g_loss": [], "status": "idle"
    }
