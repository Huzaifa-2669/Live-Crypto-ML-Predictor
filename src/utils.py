"""Utility functions for ETL quality checks and data versioning."""
import logging
import os
from pathlib import Path
import pandas as pd
from typing import Any, Dict

logger = logging.getLogger(__name__)


def quality_check(df: pd.DataFrame) -> None:
    """Raise if quality checks fail (null ratio > 1% or missing required columns)."""
    null_ratio = df.isnull().mean().mean()
    if null_ratio > 0.01:
        raise ValueError("Data quality check failed: >1% nulls")
    required_cols = ['date', 'rate', 'volume', 'cap']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Schema validation failed: missing required columns")


def load_and_version(processed_path: str, remote_path: str = "btc_processed.csv") -> None:
    """Upload processed data to MinIO via DVC."""
    # Defer imports to avoid hard dependency if not installed yet
    from minio import Minio
    processed = Path(processed_path)
    if not processed.exists():
        raise FileNotFoundError(f"{processed_path} not found")
    client = Minio(
        "localhost:9000",
        access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "admin12345"),
        secure=False,
    )
    bucket = os.getenv("MINIO_BUCKET", "mlops-data")
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info("Created bucket %s", bucket)
    client.fput_object(bucket, remote_path, str(processed))
    logger.info("Uploaded %s to bucket %s as %s", processed, bucket, remote_path)
    # DVC add/push
    os.system(f'dvc add {processed}')
    os.system('dvc push')
    logger.info("DVC add/push completed for %s", processed)


def push_context(context: Dict[str, Any], key: str, value: Any) -> None:
    """Helper to push to Airflow XCom-style context if available."""
    ti = context.get('task_instance')
    if ti:
        ti.xcom_push(key=key, value=value)
