import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data.preprocess import CATEGORICAL, NUMERICAL, clean_data
from src.features.pipeline import build_pipeline, build_preprocessor

FEATURES = NUMERICAL + CATEGORICAL


def test_preprocessor_removes_nans_and_scales(raw_df):
    X = clean_data(raw_df)[FEATURES]
    out = build_preprocessor().fit_transform(X)
    out = out.toarray() if hasattr(out, "toarray") else out
    assert not np.isnan(out).any()
    # numeric columns come first and are standardised (mean ~ 0)
    assert abs(out[:, : len(NUMERICAL)].mean()) < 1e-6


def test_pipeline_fits_and_predicts_probabilities(raw_df):
    df = clean_data(raw_df)
    pipe = build_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(df[FEATURES], df["target"])
    proba = pipe.predict_proba(df[FEATURES])
    assert proba.shape == (len(df), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_handles_unseen_category_and_missing(raw_df):
    df = clean_data(raw_df)
    pipe = build_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(df[FEATURES], df["target"])
    new = df[FEATURES].head(2).copy()
    new.loc[0, "cp"] = 99.0          # category never seen in training
    new.loc[1, "chol"] = np.nan      # missing numeric
    assert len(pipe.predict(pd.DataFrame(new))) == 2
