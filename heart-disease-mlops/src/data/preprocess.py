"""Clean the raw heart disease data."""
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/heart_disease_raw.csv")
CLEAN_PATH = Path("data/processed/heart_disease_clean.csv")

CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Binarize target, drop duplicates, validate columns. NaNs are kept on purpose."""
    df = df.copy()

    expected = set(CATEGORICAL + NUMERICAL + ["target"])
    missing_cols = expected - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # 0 = no disease, 1-4 = disease  ->  0 / 1
    df["target"] = (df["target"] > 0).astype(int)

    df = df.drop_duplicates().reset_index(drop=True)
    return df


def run(raw_path: Path = RAW_PATH, out_path: Path = CLEAN_PATH) -> pd.DataFrame:
    df = clean_data(pd.read_csv(raw_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Cleaned data saved to {out_path} | shape={df.shape}")
    print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])
    return df


if __name__ == "__main__":
    run()
