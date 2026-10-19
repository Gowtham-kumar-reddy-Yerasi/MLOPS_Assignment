import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.features.pipeline import build_pipeline
from src.models.train import load_xy

import joblib

SAMPLE = {"age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
          "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
          "ca": 0, "thal": 6}


@pytest.fixture
def client(tmp_path, monkeypatch, clean_csv):
    X, y = load_xy(clean_csv)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    model_file = tmp_path / "best_model.joblib"
    joblib.dump(pipe, model_file)
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    from src.api.app import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0


def test_predict_accepts_missing_optional_fields(client):
    payload = {k: v for k, v in SAMPLE.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=payload).status_code == 200


def test_predict_rejects_invalid_input(client):
    bad = dict(SAMPLE, age=-5)
    assert client.post("/predict", json=bad).status_code == 422
    missing = {k: v for k, v in SAMPLE.items() if k != "chol"}
    assert client.post("/predict", json=missing).status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text


def test_predict_503_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.joblib"))
    from src.api.app import app
    with TestClient(app) as c:
        assert c.post("/predict", json=SAMPLE).status_code == 503
