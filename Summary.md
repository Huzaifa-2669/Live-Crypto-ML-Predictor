# Project Brief
- Goal: Real-time BTC next-hour predictor with automated ETL/training, model registry, deployment, and monitoring.
- Stack: Airflow for orchestration; LiveCoinWatch API for data; MinIO + DVC for data storage/versioning; MLflow/Dagshub for experiment tracking and model registry; FastAPI service with Prometheus/Grafana monitoring; Dockerized deployment; GitHub Actions/CML for CI/CD (to be set later).

# Current State (local)
- Repo scaffolded: src (extract/transform/train/serve/utils), airflow/dags/etl_dag.py, data dirs, tests placeholder.
- `.env.minio` uses bucket `mlops-data` (localhost; admin/admin12345); MinIO running and test upload succeeded.
- DVC initialized with local cache dir and remote `minio` (s3://mlops-data @ http://localhost:9000); processed data DVC-tracked and pushed.
- Extract run fetched 30 days of BTC history (101 rows) to data/raw/btc_history.csv. Transform produced data/processed/btc_processed.csv (97 rows) and data/report.html. Train run (local MLflow) achieved rmse≈2031.7, mae≈1647.1, r2≈0.152.
- Python 3.12 env with core deps installed (dvc-s3, requests, pandas, fastapi, etc.); MLflow currently using local filesystem backend.

# Near-Term Focus
- Point MLflow to Dagshub (set MLFLOW_* env vars) and rerun training to log/register model there.
- Hook Airflow DAG into Airflow instance and test end-to-end ETL/train.
- Add serving/monitoring wiring once a registered model exists.
- CI/CD setup after pipeline stabilizes.
