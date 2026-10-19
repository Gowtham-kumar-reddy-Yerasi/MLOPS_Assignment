"""Download the UCI Heart Disease (Cleveland) dataset and save as CSV with headers.

Run from the project root:  python -m src.data.download_data
Fallback if the UCI URL is down: pip install ucimlrepo; fetch_ucirepo(id=45)
"""
import io
import time
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
RAW_PATH = Path("data/raw/heart_disease_raw.csv")


def download_data(url: str = URL, out_path: Path = RAW_PATH,
                  retries: int = 3) -> pd.DataFrame:
    """Fetch the raw file (with retries), attach column names, save CSV."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # fail loudly if the download fails
            break
        except requests.RequestException as exc:
            last_error = exc
            print(f"Download attempt {attempt}/{retries} failed: {exc}")
            time.sleep(2 * attempt)
    else:
        raise RuntimeError(f"Could not download dataset: {last_error}")

    df = pd.read_csv(io.StringIO(response.text), header=None,
                     names=COLUMNS, na_values="?")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} columns to {out_path}")
    return df


if __name__ == "__main__":
    download_data()
