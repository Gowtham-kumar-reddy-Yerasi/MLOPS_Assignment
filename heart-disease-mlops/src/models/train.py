"""Train, tune and compare classifiers; track every run with MLflow.

Run from the project root:  python -m src.models.train
View results:               mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)

from src.data.preprocess import CATEGORICAL, NUMERICAL
from src.features.pipeline import build_pipeline

matplotlib.use("Agg")  # save plots without opening a window

DATA_PATH = Path("data/processed/heart_disease_clean.csv")
MODEL_PATH = Path("models/best_model.joblib")
METADATA_PATH = Path("models/model_metadata.json")
RESULTS_PATH = Path("reports/model_comparison.csv")
FIG_DIR = Path("reports/figures")

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "heart-disease-classification"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
FEATURES = NUMERICAL + CATEGORICAL
SCORING = {"accuracy": "accuracy", "precision": "precision",
           "recall": "recall", "roc_auc": "roc_auc"}

# name -> (estimator, hyperparameter grid). "model__" targets the pipeline's model step.
MODELS = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000),
        {"model__C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 300],
         "model__max_depth": [3, 5, None],
         "model__min_samples_leaf": [1, 3, 5]},
    ),
}


def load_xy(path: Path = DATA_PATH):
    """Features as float (consistent with what the API sends) and the target."""
    df = pd.read_csv(path)
    return df[FEATURES].astype(float), df["target"]


def evaluate_on_test(pipe, X_test, y_test) -> dict:
    """Metrics on the untouched test set."""
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred),
        "test_recall": recall_score(y_test, pred),
        "test_roc_auc": roc_auc_score(y_test, proba),
    }


def make_plots(pipe, name, X_test, y_test, fig_dir: Path) -> list:
    """Save ROC curve and confusion matrix for one model; return file paths."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="chance")
    ax.set_title(f"ROC curve: {name}")
    ax.legend()
    p = fig_dir / f"roc_{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, ax=ax)
    ax.set_title(f"Confusion matrix: {name}")
    p = fig_dir / f"confusion_{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths


def save_comparison_roc(fitted, X_test, y_test, fig_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, pipe in fitted.items():
        RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="chance")
    ax.set_title("ROC curves (test set)")
    ax.legend()
    p = fig_dir / "roc_curves.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def log_candidates(name: str, search: GridSearchCV) -> None:
    """One nested MLflow run per hyperparameter combination tried."""
    res = search.cv_results_
    for i, params in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"{name}-candidate-{i}", nested=True):
            mlflow.log_params({k.replace("model__", ""): v for k, v in params.items()})
            mlflow.log_metric("cv_roc_auc", float(res["mean_test_score"][i]))
            mlflow.log_metric("cv_roc_auc_std", float(res["std_test_score"][i]))


def train_all(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH,
              results_path: Path = RESULTS_PATH, fig_dir: Path = FIG_DIR,
              metadata_path: Path = METADATA_PATH, models: dict = None,
              tracking_uri: str = TRACKING_URI):
    models = models or MODELS
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_xy(data_path)
    # stratify keeps the same class ratio in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted = [], {}
    for name, (estimator, grid) in models.items():
        print(f"\n=== {name} ===")
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"model_type": name,
                             "dataset": "UCI Heart Disease (Cleveland)"})
            mlflow.log_params({
                "model_name": name, "test_size": TEST_SIZE, "cv_folds": CV_FOLDS,
                "random_state": RANDOM_STATE, "n_train": len(X_train),
                "n_test": len(X_test), "tuning_metric": "roc_auc",
            })

            search = GridSearchCV(build_pipeline(estimator), grid, cv=cv,
                                  scoring="roc_auc", n_jobs=-1)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            print("best params:", search.best_params_)
            log_candidates(name, search)
            mlflow.log_params({f"best_{k.replace('model__', '')}": v
                               for k, v in search.best_params_.items()})

            cv_res = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING)
            row = {"model": name, "best_params": str(search.best_params_)}
            for metric in SCORING:
                row[f"cv_{metric}"] = float(cv_res[f"test_{metric}"].mean())
                row[f"cv_{metric}_std"] = float(cv_res[f"test_{metric}"].std())
            row.update({k: float(v)
                        for k, v in evaluate_on_test(best, X_test, y_test).items()})
            mlflow.log_metrics({k: v for k, v in row.items()
                                if k.startswith(("cv_", "test_"))})

            for plot in make_plots(best, name, X_test, y_test, fig_dir):
                mlflow.log_artifact(str(plot), artifact_path="plots")
            mlflow.sklearn.log_model(best, name="model",
                                     serialization_format="cloudpickle")

            rows.append(row)
            fitted[name] = best

    results = pd.DataFrame(rows).round(4)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    print("\n", results[["model", "cv_accuracy", "cv_precision", "cv_recall",
                         "cv_roc_auc", "test_roc_auc"]].to_string(index=False))

    # choose by cross-validation score, NOT by test score
    best_row = results.sort_values("cv_roc_auc", ascending=False).iloc[0]
    best_name = best_row["model"]
    print(f"\nSelected model: {best_name}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], model_path)
    metadata = {
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "cv_roc_auc": float(best_row["cv_roc_auc"]),
        "test_roc_auc": float(best_row["test_roc_auc"]),
        "best_params": best_row["best_params"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved pipeline to {model_path} and metadata to {metadata_path}")

    roc_path = save_comparison_roc(fitted, X_test, y_test, fig_dir)
    with mlflow.start_run(run_name="model_comparison"):
        mlflow.set_tag("selected_model", best_name)
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metric("selected_cv_roc_auc", float(best_row["cv_roc_auc"]))
        mlflow.log_metric("selected_test_roc_auc", float(best_row["test_roc_auc"]))
        mlflow.log_artifact(str(results_path), artifact_path="reports")
        mlflow.log_artifact(str(roc_path), artifact_path="plots")
        mlflow.log_artifact(str(metadata_path), artifact_path="reports")
        heatmap = fig_dir / "correlation_heatmap.png"
        if heatmap.exists():
            mlflow.log_artifact(str(heatmap), artifact_path="eda")

    return results, fitted[best_name]


if __name__ == "__main__":
    train_all()
