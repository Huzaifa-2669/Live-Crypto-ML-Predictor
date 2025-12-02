from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging
import os
import sys

logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from extract import extract_data
from utils import quality_check, load_and_version, push_context
from transform import transform_data
from train import train_model

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def extract_task(**context):
    df = extract_data()
    push_context(context, "df", df)
    logger.info("Extract task produced %d rows", len(df))


def quality_task(**context):
    ti = context["task_instance"]
    df = ti.xcom_pull(key="df", task_ids="extract")
    quality_check(df)
    push_context(context, "df_qc", df)
    logger.info("Quality check passed for %d rows", len(df))


def transform_task(**context):
    ti = context["task_instance"]
    df = ti.xcom_pull(key="df_qc", task_ids="quality_check")
    processed = transform_data(df)
    push_context(context, "processed_df", processed)
    logger.info("Transform produced %d rows with features", len(processed))


def load_task(**context):
    load_and_version("data/processed/btc_processed.csv", "btc_processed.csv")
    logger.info("Load/version completed for processed dataset")


with DAG(
    "etl_dag",
    default_args=default_args,
    schedule_interval="@hourly",
    catchup=False,
) as dag:
    extract = PythonOperator(task_id="extract", python_callable=extract_task)
    quality = PythonOperator(task_id="quality_check", python_callable=quality_task, provide_context=True)
    transform = PythonOperator(task_id="transform", python_callable=transform_task, provide_context=True)
    load_version = PythonOperator(task_id="load_version", python_callable=load_task, provide_context=True)
    train = PythonOperator(task_id="train", python_callable=train_model)

    extract >> quality >> transform >> load_version >> train
