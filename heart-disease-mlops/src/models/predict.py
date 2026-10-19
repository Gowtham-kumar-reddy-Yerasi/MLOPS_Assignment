"""Offline inference: load the saved pipeline and score a JSON patient record.

Run:  python -m src.models.predict scripts/sample_request.json
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.data.preprocess import CATEGORICAL, NUMERICAL

MODEL_PATH = Path("models/best_model.joblib")
FEATURES = NUMERICAL + CATEGORICAL


def predict_record(record: dict, model_path: Path = MODEL_PATH) -> dict:
    model = joblib.load(model_path)
    df = pd.DataFrame([record])[FEATURES].astype(float)
    proba = model.predict_proba(df)[0]
    classes = list(model.classes_)
    pred = int(model.predict(df)[0])
    return {"prediction": pred, "confidence": float(proba[classes.index(pred)]),
            "probability_disease": float(proba[classes.index(1)])}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "scripts/sample_request.json"
    print(predict_record(json.loads(Path(path).read_text())))
