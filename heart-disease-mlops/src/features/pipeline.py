"""Feature engineering: build the preprocessing + model pipeline.

Everything that learns from data (median, mean, std, categories) lives inside
a scikit-learn Pipeline, so it is fitted on TRAINING data only (no leakage)
and saved together with the model (full reproducibility).
"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.preprocess import CATEGORICAL, NUMERICAL


def build_preprocessor() -> ColumnTransformer:
    """Numeric: median-impute then scale. Categorical: mode-impute then one-hot."""
    numeric_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, NUMERICAL),
        ("cat", categorical_steps, CATEGORICAL),
    ])


def build_pipeline(model) -> Pipeline:
    """Full pipeline: preprocessing followed by the given classifier."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])
