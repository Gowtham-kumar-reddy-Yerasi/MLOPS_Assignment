"""Shared test fixtures: a small synthetic dataset with the real schema."""
import numpy as np
import pandas as pd
import pytest


def make_raw_df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, n)  # raw target is 0-4 severity
    sick = (y > 0).astype(int)
    df = pd.DataFrame({
        "age": rng.normal(54, 9, n) + sick * 3,
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": rng.integers(1, 5, n).astype(float),
        "trestbps": rng.normal(131, 17, n),
        "chol": rng.normal(246, 50, n),
        "fbs": rng.integers(0, 2, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "thalach": rng.normal(150, 22, n) - sick * 15,
        "exang": rng.integers(0, 2, n).astype(float),
        "oldpeak": np.abs(rng.normal(1, 1, n)) + sick,
        "slope": rng.integers(1, 4, n).astype(float),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
        "target": y,
    })
    df.loc[[3, 7], "ca"] = np.nan
    df.loc[[5], "thal"] = np.nan
    return df


@pytest.fixture
def raw_df():
    return make_raw_df()


@pytest.fixture
def clean_csv(tmp_path, raw_df):
    from src.data.preprocess import clean_data
    path = tmp_path / "clean.csv"
    clean_data(raw_df).to_csv(path, index=False)
    return path
