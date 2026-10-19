"""FastAPI service that serves the heart disease model.

Endpoints:  GET /health   POST /predict   GET /metrics (Prometheus)
Run locally:  uvicorn src.api.app:app --reload
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, ConfigDict, Field

from src.data.preprocess import CATEGORICAL, NUMERICAL

FEATURES = NUMERICAL + CATEGORICAL
KNOWN_PATHS = {"/", "/health", "/predict", "/metrics", "/docs", "/openapi.json"}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("heart_api")

# ---- Prometheus metrics ----
REQUEST_COUNT = Counter("api_requests_total", "Total API requests",
                        ["method", "path", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds",
                            "Request latency in seconds", ["path"])
PREDICTION_COUNT = Counter("model_predictions_total",
                           "Predictions served", ["label"])


class PatientFeatures(BaseModel):
    """One patient record. ca and thal may be omitted (imputed by the pipeline)."""
    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6}})

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting BP (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=30, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST")
    ca: Optional[int] = Field(None, ge=0, le=3, description="Major vessels (0-3)")
    thal: Optional[Literal[3, 6, 7]] = Field(
        None, description="3 normal, 6 fixed defect, 7 reversible defect")


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float


def load_model():
    path = Path(os.getenv("MODEL_PATH", "models/best_model.joblib"))
    meta_path = path.with_name("model_metadata.json")
    try:
        model = joblib.load(path)
    except Exception:
        logger.exception("Could not load model from %s", path)
        return None, {}
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        trained_with = meta.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            logger.warning("scikit-learn mismatch: trained with %s, serving with %s",
                           trained_with, sklearn.__version__)
    logger.info("Model loaded from %s (%s)", path, meta.get("model_name", "unknown"))
    return model, meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.meta = load_model()
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start = time.perf_counter()
    path = request.url.path if request.url.path in KNOWN_PATHS else "other"
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        duration = time.perf_counter() - start
        REQUEST_COUNT.labels(request.method, path, str(status)).inc()
        REQUEST_LATENCY.labels(path).observe(duration)
        logger.info("request method=%s path=%s status=%s duration_ms=%.1f",
                    request.method, path, status, duration * 1000)


@app.get("/")
def root():
    return {"service": "heart-disease-api", "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health(request: Request):
    meta = request.app.state.meta
    return {"status": "ok", "model_loaded": request.app.state.model is not None,
            "model_name": meta.get("model_name")}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df = pd.DataFrame([patient.model_dump()])[FEATURES].astype(float)
    proba = model.predict_proba(df)[0]
    classes = list(model.classes_)
    pred = int(model.predict(df)[0])
    result = PredictionResponse(
        prediction=pred,
        label="heart disease likely" if pred == 1 else "heart disease unlikely",
        confidence=round(float(proba[classes.index(pred)]), 4),
        probability_disease=round(float(proba[classes.index(1)]), 4),
    )
    PREDICTION_COUNT.labels(str(pred)).inc()
    # Log only the outcome, never the patient's medical values (privacy).
    logger.info("prediction=%s confidence=%.3f", pred, result.confidence)
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
