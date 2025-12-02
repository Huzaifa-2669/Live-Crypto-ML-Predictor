# Project Brief
- Goal: Real-time BTC next-hour predictor with automated ETL/training, model registry, deployment, and monitoring.
- Stack: Airflow for orchestration; LiveCoinWatch API for data; MinIO + DVC for data storage/versioning; MLflow/Dagshub for experiment tracking and model registry; FastAPI service with Prometheus/Grafana monitoring; Dockerized deployment; GitHub Actions/CML for CI/CD (to be set later).

# Current State (local)
- Repo has instructions, MinIO compose, and a MinIO test upload script; no code scaffolding for ETL/model yet.
- `.env.minio` updated to use bucket `mlops-data` (endpoint localhost; admin/admin12345).
- Python 3.12 available; `venv/` exists; deps installation in progress.
- MinIO container started via docker-compose; test upload succeeded (samples/hello.txt in bucket mlops-data).
- DVC initialized with local cache dir and remote `minio` pointing to s3://mlops-data @ http://localhost:9000 (admin/admin12345).

# Near-Term Focus
- Validate env/venv and dependencies.
- Confirm dependency installation status in venv (minio installed; check others).
- Scaffold project directories and baseline files for ETL/training/serving before CI/CD.
