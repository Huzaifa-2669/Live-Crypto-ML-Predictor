# Live-Crypto-ML-Predictor

End-to-end MLOps project to predict the next-hour BTC price from LiveCoinWatch data. The system ingests live data, versions it, trains and tracks models, serves predictions with monitoring, and provides CI/CD hooks.

## Architecture Snapshot
- **Source:** LiveCoinWatch API (current BTC rate/volume/cap).
- **Orchestration:** Airflow DAG (`etl_dag.py`) scheduled hourly.
- **Storage/Versioning:** DVC + MinIO (S3-compatible) for processed data; optional DagsHub DVC remote.
- **Experiment & Model Registry:** MLflow pointing to DagsHub (model `btc_predictor`).
- **Serving:** FastAPI (`src/serve.py`) loading model via `MODEL_URI`; Prometheus metrics at `/metrics`.
- **Monitoring:** Prometheus + Grafana (compose + dashboard provided).
- **CI:** GitHub Actions (`.github/workflows/ci.yml`) for lint/test/build.
- **Containers:** Dockerfile for serving; docker-compose for monitoring.

## Repository Layout
- `src/` – core logic (`extract.py`, `transform.py`, `train.py`, `serve.py`, `utils.py`)
- `airflow/dags/etl_dag.py` – extract → quality → transform → load/version → train
- `data/` – raw/processed (DVC-tracked)
- `tests/` – unit tests; `tests/conftest.py` sets project root on `sys.path`
- `Dockerfile` – serving image
- `prometheus.yml`, `docker-compose.monitoring.yml`, `grafana-dashboard.json` – monitoring stack
- `.github/workflows/ci.yml` – CI (lint/test/build placeholder for pipeline)
- `.airflow/` – local Airflow state (created when running Airflow)

## Prerequisites
- Python 3.12
- Docker (for MinIO, serving image, monitoring stack)
- LiveCoinWatch API key
- DagsHub account/repo (e.g., `i222669/mlops-rps`) for MLflow/DVC

## Environment Variables (core)
```
LIVE_COIN_WATCH_API_KEY=...
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=admin12345
MINIO_BUCKET=mlops-data
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<user>
MLFLOW_TRACKING_PASSWORD=<token>
DVC_BIN=$PWD/venv/bin/dvc          # or airflow-venv/bin/dvc when running Airflow
AIRFLOW_HOME=$PWD/.airflow         # when running Airflow locally
```

## Quickstart (local dev)
1) **Setup venv & deps**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```
2) **Start MinIO**
   ```bash
   docker compose -f docker-compose.minio.yml up -d
   ```
3) **Configure DVC to MinIO**
   ```bash
   dvc remote add -d minio s3://mlops-data
   dvc remote modify minio endpointurl http://localhost:9000
   dvc remote modify minio access_key_id admin
   dvc remote modify minio secret_access_key admin12345
   ```
4) **Run ETL manually**
   ```bash
   python src/extract.py
   python src/transform.py
   python src/train.py   # logs to MLflow (DagsHub), registers model
   ```
5) **Serve locally**
   ```bash
   MODEL_URI="models:/btc_predictor/1" uvicorn src.serve:app --host 0.0.0.0 --port 8000
   # endpoints: / , /health , /predict , /metrics
   ```
6) **Monitoring stack (Prometheus + Grafana)**
   ```bash
   docker compose -f docker-compose.monitoring.yml up -d
   # Prometheus: http://localhost:9090 (if running service on host, set target to host.docker.internal:8000 on macOS/Windows)
   # Grafana: http://localhost:3000 (import grafana-dashboard.json, add Prometheus datasource)
   ```

## Airflow (local)
```bash
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$AIRFLOW_HOME/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__LOGGING__BASE_LOG_FOLDER="$AIRFLOW_HOME/logs"
export AIRFLOW__LOGGING__DAG_PROCESSOR_MANAGER_LOG_LOCATION="$AIRFLOW_HOME/logs/dag_processor_manager.log"
ln -sf "$PWD/airflow/dags/etl_dag.py" "$AIRFLOW_HOME/dags/etl_dag.py"

./airflow-venv/bin/airflow db migrate     # adjust to your Airflow venv
./airflow-venv/bin/airflow standalone     # prints admin password, starts webserver+scheduler
```
Log in at http://localhost:8080, enable `etl_dag`, trigger a run. Ensure env vars above (API key, MLflow, MinIO, DVC_BIN) are set in the shells running webserver/scheduler. If imports are slow, set `AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=120`.

## Serving in Docker
```bash
docker build -t mlops-rps:latest .
docker run -e MODEL_URI="models:/btc_predictor/1" -p 8000:8000 mlops-rps:latest
```

## Monitoring (details)
- Prometheus: configured via `prometheus.yml` to scrape `/metrics`.
- Grafana: import `grafana-dashboard.json` for `requests_total`, latency quantiles, and `drift_ratio`.

## CI/CD
- GitHub Actions (`.github/workflows/ci.yml`):
  - `lint-test`: installs requirements, runs flake8 and pytest.
  - `build-serve`: builds Docker image.
  - `pipeline-check`: placeholder (extend to mock ETL/train).
- Branch protection (suggested): require PR + passing checks (`lint-test`, `build-serve`, `pipeline-check`) on main/dev/test; optionally require up-to-date merges.

## DagsHub Integration
- Git remote: `https://dagshub.com/<user>/<repo>.git`
- DVC remote (optional): `https://dagshub.com/<user>/<repo>.dvc` (set `user`/`password` token with `dvc remote modify ... --local`)
- MLflow tracking: `https://dagshub.com/<user>/<repo>.mlflow` (pinned to mlflow 2.13.1 for registry compatibility)
- Model registry: `btc_predictor` (serve with `models:/btc_predictor/<stage|version>`)

## Notes & Troubleshooting
- Airflow timeouts: lazy imports applied in DAG tasks; increase DAGBAG import timeout if needed.
- Missing deps in Airflow: install pandas, ydata-profiling, pyarrow in the Airflow venv.
- MinIO access: ensure `MINIO_ENDPOINT` and creds are exported in Airflow runtime.
- DVC auth to DagsHub: set `user`/`password` via `dvc remote modify <remote> --local`.
- Serving model load errors: verify `MODEL_URI` and MLflow creds; ensure model is registered in DagsHub.
