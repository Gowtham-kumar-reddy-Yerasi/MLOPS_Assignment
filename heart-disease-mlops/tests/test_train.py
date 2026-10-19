import json

import joblib
from sklearn.linear_model import LogisticRegression

from src.models.train import load_xy, train_all


def test_load_xy_shapes(clean_csv):
    X, y = load_xy(clean_csv)
    assert X.shape[1] == 13
    assert len(X) == len(y)


def test_train_all_creates_artifacts(tmp_path, monkeypatch, clean_csv):
    monkeypatch.chdir(tmp_path)  # keep mlruns/ out of the repo
    models = {"logreg": (LogisticRegression(max_iter=500), {"model__C": [0.1, 1]})}
    results, best = train_all(
        data_path=clean_csv,
        model_path=tmp_path / "m" / "model.joblib",
        results_path=tmp_path / "res.csv",
        fig_dir=tmp_path / "figs",
        metadata_path=tmp_path / "m" / "meta.json",
        models=models,
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )
    assert (tmp_path / "m" / "model.joblib").exists()
    assert (tmp_path / "res.csv").exists()
    assert (tmp_path / "figs" / "roc_curves.png").exists()
    meta = json.loads((tmp_path / "m" / "meta.json").read_text())
    assert meta["model_name"] == "logreg"
    assert 0.0 <= results.loc[0, "cv_roc_auc"] <= 1.0
    reloaded = joblib.load(tmp_path / "m" / "model.joblib")
    X, _ = load_xy(clean_csv)
    assert len(reloaded.predict(X)) == len(X)
