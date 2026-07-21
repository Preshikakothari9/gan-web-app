"""
API-level tests, run against Flask's test client (no real server, no
network). Deliberately avoid calling /train for real — that would kick off
an actual MNIST download / training loop, which is slow and needs network
access. Instead we check that training routes accept/reject requests
correctly and rely on generate/status/health for functional coverage.
"""


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["device"] in ("cpu", "cuda")
    assert body["auth_required"] is False


def test_models_status_shape(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.get_json()
    assert "image_trained" in body
    assert "text_trained" in body
    assert isinstance(body["image_trained"], bool)
    assert isinstance(body["text_trained"], bool)


def test_unknown_api_route_returns_json_404(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    assert r.is_json
    assert r.get_json()["error"] == "Not found"


def test_unknown_non_api_route_falls_back_to_spa_shell(client):
    r = client.get("/some/client/route")
    assert r.status_code == 200
    assert b"<html" in r.data.lower() or b"<!doctype" in r.data.lower()


def test_image_generate_untrained_returns_valid_png(client):
    r = client.get("/api/image/generate?n=4")
    assert r.status_code == 200
    body = r.get_json()
    assert body["trained"] is False
    assert body["image"].startswith("data:image/png;base64,")


def test_text_generate_untrained_returns_sentences(client):
    r = client.get("/api/text/generate?n=3")
    assert r.status_code == 200
    body = r.get_json()
    assert body["trained"] is False
    assert len(body["sentences"]) == 3
    assert all(isinstance(s, str) for s in body["sentences"])


def test_image_stop_is_idempotent_when_not_running(client):
    r = client.post("/api/image/stop")
    assert r.status_code == 200
    assert r.get_json()["message"] == "Stopped"


def test_image_status_default_shape(client):
    r = client.get("/api/image/status")
    assert r.status_code == 200
    body = r.get_json()
    for key in ("running", "epoch", "total", "d_loss", "g_loss", "status"):
        assert key in body


def test_image_train_rejects_when_already_running(client, monkeypatch):
    import app as flask_app_module
    flask_app_module.state["image"]["running"] = True
    r = client.post("/api/image/train", json={"epochs": 1})
    assert r.status_code == 400
    assert "Already training" in r.get_json()["error"]


# ── Auth gating ──────────────────────────────────────────────────────────

def test_train_open_when_no_api_key_configured(client, monkeypatch):
    # With API_KEY unset (the `client` fixture), training endpoints don't
    # require a header at all. We don't let it actually train (would try to
    # download MNIST); we just confirm it's not rejected for auth reasons.
    import app as flask_app_module

    started = {}

    def fake_thread(target, args, daemon):
        started["called"] = True
        class _Noop:
            def start(self_inner):
                pass
        return _Noop()

    monkeypatch.setattr(flask_app_module.threading, "Thread", fake_thread)
    r = client.post("/api/image/train", json={"epochs": 1})
    assert r.status_code == 200
    assert started.get("called") is True


def test_train_rejected_without_key_when_api_key_configured(authed_client):
    r = authed_client.post("/api/image/train", json={"epochs": 1})
    assert r.status_code == 401


def test_train_rejected_with_wrong_key(authed_client):
    r = authed_client.post(
        "/api/image/train",
        json={"epochs": 1},
        headers={"X-API-Key": "wrong-key"},
    )
    assert r.status_code == 401


def test_train_accepted_with_correct_key(authed_client, monkeypatch):
    import app as flask_app_module

    def fake_thread(target, args, daemon):
        class _Noop:
            def start(self_inner):
                pass
        return _Noop()

    monkeypatch.setattr(flask_app_module.threading, "Thread", fake_thread)
    r = authed_client.post(
        "/api/image/train",
        json={"epochs": 1},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert r.status_code == 200


def test_stop_also_gated_when_api_key_configured(authed_client):
    r = authed_client.post("/api/image/stop")
    assert r.status_code == 401
    r2 = authed_client.post("/api/image/stop", headers={"X-API-Key": "test-secret-key"})
    assert r2.status_code == 200


def test_generate_and_status_stay_open_even_with_api_key_configured(authed_client):
    # Read-only endpoints shouldn't require the key — only the endpoints
    # that actually spend compute (train/stop) are gated.
    assert authed_client.get("/api/health").status_code == 200
    assert authed_client.get("/api/models").status_code == 200
    assert authed_client.get("/api/image/generate?n=2").status_code == 200
    assert authed_client.get("/api/text/generate?n=2").status_code == 200
