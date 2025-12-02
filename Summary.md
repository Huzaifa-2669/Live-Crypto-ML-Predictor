# Project Brief
- Goal: Real-time BTC next-hour predictor with automated ETL/training, model registry, deployment, and monitoring.
- Stack: Airflow for orchestration; LiveCoinWatch API for data; MinIO + DVC for data storage/versioning; MLflow/Dagshub for experiment tracking and model registry; FastAPI service with Prometheus/Grafana monitoring; Dockerized deployment; GitHub Actions/CML for CI/CD (to be set later).

# Current State (local)
- Repo scaffolded: src (extract/transform/train/serve/utils), airflow/dags/etl_dag.py, data dirs, tests placeholder; README added; Dockerfile for serving.
- `.env.minio` uses bucket `mlops-data` (localhost; admin/admin12345); MinIO running and test upload succeeded.
- DVC initialized with local cache dir and remote `minio` (s3://mlops-data @ http://localhost:9000); processed data tracked/pushed; processed data/report also uploaded to DagsHub bucket.
- Extract run fetched 30 days of BTC history (101 rows) to data/raw/btc_history.csv. Transform produced data/processed/btc_processed.csv (97 rows) and data/report.html. Train run (MLflow@DagsHub, mlflow 2.13.1) achieved rmse≈2031.7, mae≈1647.1, r2≈0.152; model registered as btc_predictor v1.
- Serving loads model via env-driven MODEL_URI; root/health/predict/metrics endpoints; Dockerfile runs uvicorn.
- Airflow in progress: DAG shows in UI; import fixes (lazy imports, schedule, no provide_context). Airflow metadata/log paths and env need to be consistent; missing deps fixed (pandas/ydata-profiling/pyarrow installed in airflow-venv). Run failures remain to be stabilized.

# Near-Term Focus
- Stabilize Airflow execution: ensure consistent AIRFLOW_HOME/log settings and env vars; increase DAG import timeout if needed; rerun etl_dag end-to-end.
- Add Prometheus/Grafana monitoring wiring for serving.
- CI/CD setup after pipeline stabilizes; consider lighter serving requirements file if needed for Docker image size.
