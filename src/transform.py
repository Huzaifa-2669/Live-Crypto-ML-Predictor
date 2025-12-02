"""Transformation step: clean and feature engineer BTC data."""
import logging
from pathlib import Path
import pandas as pd

# Support both ydata-profiling (new) and pandas-profiling (legacy)
try:
    from ydata_profiling import ProfileReport  # preferred
except ImportError:
    from pandas_profiling import ProfileReport  # fallback

logger = logging.getLogger(__name__)


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    df["lag_1"] = df["rate"].shift(1)
    df["rolling_mean_5"] = df["rate"].rolling(window=5).mean()
    df["hour"] = df["date"].dt.hour
    df = df.dropna()

    profile = ProfileReport(df, title="Data Profile", minimal=True)
    report_path = Path("data/report.html")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_file(report_path)
    processed_path = Path("data/processed/btc_processed.csv")
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)
    logger.info("Transformed data shape: %s; saved to %s and report to %s", df.shape, processed_path, report_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    raw_path = Path("data/raw/btc_history.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run extract first.")
    df = pd.read_csv(raw_path)
    out = transform_data(df)
    print(f"Processed shape: {out.shape}")
    print("Saved processed data to data/processed/btc_processed.csv and report to data/report.html")
