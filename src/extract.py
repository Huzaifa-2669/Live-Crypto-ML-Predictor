"""Extraction step: fetch BTC data from LiveCoinWatch."""
import logging
import os
import time
from pathlib import Path
import requests
import pandas as pd

logger = logging.getLogger(__name__)


def extract_data() -> pd.DataFrame:
    api_key = os.getenv("LIVE_COIN_WATCH_API_KEY")
    headers = {"content-type": "application/json", "x-api-key": api_key}
    raw_path = Path("data/raw/btc_history.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    now_ms = int(time.time() * 1000)

    if not raw_path.exists():
        logger.info("No existing raw file found. Fetching 30 days of history to %s", raw_path)
        payload = {
            "currency": "USD",
            "code": "BTC",
            "start": int((time.time() - 30 * 24 * 3600) * 1000),
            "end": now_ms,
            "meta": False,
        }
        resp = requests.post(
            "https://api.livecoinwatch.com/coins/single/history",
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("history", [])
        df = pd.DataFrame(data)
        df.to_csv(raw_path, index=False)
        logger.info("Wrote %d rows to %s", len(df), raw_path)
        return df

    payload = {"currency": "USD", "code": "BTC", "meta": False}
    resp = requests.post(
        "https://api.livecoinwatch.com/coins/single",
        json=payload,
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    df_existing = pd.read_csv(raw_path)
    logger.info("Loaded existing raw file %s with %d rows", raw_path, len(df_existing))
    new_row = {
        "date": now_ms,
        "rate": data.get("rate"),
        "volume": data.get("volume"),
        "cap": data.get("cap"),
    }
    df_new = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
    df_new.to_csv(raw_path, index=False)
    logger.info("Appended new row; new total rows: %d", len(df_new))
    return df_new


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    df = extract_data()
    print(f"Wrote {len(df)} rows to data/raw/btc_history.csv")
