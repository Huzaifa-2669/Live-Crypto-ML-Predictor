# Live-Crypto-ML-Predictor

Real-time MLOps pipeline to predict next-hour BTC price using LiveCoinWatch data. Stack: Airflow, DVC+MinIO, MLflow (DagsHub), FastAPI + Prometheus, Docker.

## Repo Layout
- `src/` – extract/transform/train/serve/utils
- `airflow/dags/etl_dag.py` – hourly ETL + train DAG
- `data/` – raw/processed (DVC-tracked)
- `tests/` – unit tests placeholder
- `Dockerfile` – serving container
- `.airflow/` – local Airflow state (created at runtime)

## Prereqs
- Python 3.12
- Docker (for MinIO/Docker builds)
- LiveCoinWatch API key
- DagsHub account/repo for MLflow/DVC (e.g., `i222669/mlops-rps`)

## Environment Variables
```
LIVE_COIN_WATCH_API_KEY=...
MINIO_ENDPOINT=http://localhost:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=admin12345
MINIO_BUCKET=mlops-data
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<user>
MLFLOW_TRACKING_PASSWORD=<token>
DVC_BIN=$PWD/venv/bin/dvc
AIRFLOW_HOME=$PWD/.airflow
```

## Quickstart (local)
1) **Setup venv & deps**  
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```
2) **Start MinIO**  
   ```
   docker compose -f docker-compose.minio.yml up -d
   ```
3) **Configure DVC (MinIO default)**  
   ```
   dvc remote add -d minio s3://mlops-data
   dvc remote modify minio endpointurl http://localhost:9000
   dvc remote modify minio access_key_id admin
   dvc remote modify minio secret_access_key admin12345
   ```
4) **Run ETL manually**  
   ```
   python src/extract.py
   python src/transform.py
   python src/train.py   # logs to MLflow (DagsHub)
   ```
5) **Serving locally**  
   ```
   MODEL_URI="models:/btc_predictor/1" uvicorn src.serve:app --host 0.0.0.0 --port 8000
   # endpoints: /, /health, /predict, /metrics
   ```

## Airflow (local)
```
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$AIRFLOW_HOME/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__LOGGING__BASE_LOG_FOLDER="$AIRFLOW_HOME/logs"
export AIRFLOW__LOGGING__DAG_PROCESSOR_MANAGER_LOG_LOCATION="$AIRFLOW_HOME/logs/dag_processor_manager.log"
ln -sf "$PWD/airflow/dags/etl_dag.py" "$AIRFLOW_HOME/dags/etl_dag.py"

airflow db migrate
airflow standalone   # prints admin password, starts webserver+scheduler
```
Then log in at http://localhost:8080, enable `etl_dag`, trigger a run.

## Docker Serving
```
docker build -t mlops-rps:latest .
docker run -e MODEL_URI="models:/btc_predictor/1" -p 8000:8000 mlops-rps:latest
```

## DagsHub Integration
- Git remote: `https://dagshub.com/<user>/<repo>.git`
- DVC remote (optional, in addition to MinIO): `https://dagshub.com/<user>/<repo>.dvc`
- MLflow tracking: `https://dagshub.com/<user>/<repo>.mlflow`

## Monitoring (later)
- Prometheus scrape: `:8000/metrics`
- Grafana dashboard for `inference_latency_seconds`, `requests_total`, `drift_ratio`

## CI/CD (later)
- GitHub Actions/CML: lint/tests on PR to dev; retrain/compare on test; build/push Docker on master.

## Notes
- MLflow pinned to 2.13.1 for DagsHub registry compatibility.
- Airflow DAG uses lazy imports to speed parse time.
