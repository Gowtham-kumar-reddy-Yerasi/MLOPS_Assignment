import numpy as np
import pytest

from src.data.preprocess import CATEGORICAL, NUMERICAL, clean_data


def test_target_is_binary(raw_df):
    out = clean_data(raw_df)
    assert set(out["target"].unique()) <= {0, 1}


def test_severity_above_zero_becomes_one(raw_df):
    raw_df.loc[0, "target"] = 3
    raw_df.loc[1, "target"] = 0
    out = clean_data(raw_df)
    assert out.loc[0, "target"] == 1
    assert out.loc[1, "target"] == 0


def test_missing_values_are_kept(raw_df):
    out = clean_data(raw_df)
    assert out["ca"].isna().sum() == 2
    assert out["thal"].isna().sum() == 1


def test_duplicates_removed(raw_df):
    doubled = np.concatenate([raw_df.values, raw_df.values[:5]])
    import pandas as pd
    df = pd.DataFrame(doubled, columns=raw_df.columns)
    assert len(clean_data(df)) == len(clean_data(raw_df))


def test_missing_column_raises(raw_df):
    with pytest.raises(ValueError, match="Missing columns"):
        clean_data(raw_df.drop(columns=["chol"]))


def test_input_not_modified(raw_df):
    before = raw_df.copy()
    clean_data(raw_df)
    assert raw_df.equals(before)


def test_feature_lists_cover_all_columns():
    assert len(CATEGORICAL) + len(NUMERICAL) == 13
