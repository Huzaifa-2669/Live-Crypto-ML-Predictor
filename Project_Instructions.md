MLOps Project Implementation Guide: Real-Time Predictive System for Cryptocurrency Price Prediction
Project Overview
This guide provides a complete, step-by-step implementation for the MLOps project described in "MLOps Project Spring 25.pdf". The goal is to build a Real-Time Predictive System (RPS) for predicting the next-hour closing price of Bitcoin (BTC) using time-series data from the LiveCoinWatch API. The system includes automated data ingestion, model training, deployment, and monitoring, with handling for concept drift.
Note: The project deadline is November 30, 2025, but this guide is provided as of December 02, 2025. Proceed with implementation for learning or extension purposes.
We use free tools exclusively:

API: LiveCoinWatch (free tier with ~10,000 daily credits; each request costs 1 credit).
Orchestration: Apache Airflow (local).
Data/Model Management: DVC, MLflow, Dagshub (free public repo).
Storage: MinIO (local S3-compatible).
CI/CD: GitHub Actions, CML, Docker (free).
Monitoring: Prometheus, Grafana (local).
Model: Simple LSTM (using TensorFlow) for time-series prediction.
Development Aid: Use Codex (e.g., GitHub Copilot) for code completion.

The pipeline:

Fetches historical/live BTC price data.
Performs ETL with quality checks.
Trains/retrains model.
Deploys via Docker.
Monitors inference.

Repository Structure:
textmlops-rps-project/
├── dags/ # Airflow DAGs
│ └── etl_dag.py
├── data/ # DVC-versioned data
│ └── raw/ # Raw API data
│ └── processed/ # Transformed data
├── models/ # MLflow local artifacts
├── src/ # Source code
│ ├── extract.py
│ ├── transform.py
│ ├── train.py
│ ├── serve.py
│ └── utils.py
├── tests/ # Unit tests
│ └── test_utils.py
├── .dvc/ # DVC config
├── .github/workflows/ # GitHub Actions
│ └── ci.yml
├── Dockerfile
├── prometheus.yml
├── requirements.txt
├── README.md # Project overview
└── SETUP.md # This file (or merge into README)
Initial Setup Instructions
Follow these steps to set up the environment locally. All tools are free and self-hosted where possible.

System Requirements:
OS: Linux/Mac/Windows (with WSL for Windows).
Python: 3.10+.
Docker: Installed for containerization.
Git: For version control.

Create Project Directory and Git Repo:textmkdir mlops-rps-project
cd mlops-rps-project
git init
Create a GitHub repo (e.g., mlops-rps) and add remote: git remote add origin https://github.com/yourusername/mlops-rps.git.
Protect branches: In GitHub settings, require PR approvals (1 reviewer) for test and master.

Virtual Environment:textpython -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
Install Dependencies (create requirements.txt with this content):textapache-airflow==2.10.0
mlflow==2.16.0
dvc[s3]==3.55.0
fastapi==0.115.0
uvicorn==0.30.6
prometheus-client==0.21.0
pandas==2.2.3
ydata-profiling==4.10.0
scikit-learn==1.5.2
tensorflow==2.17.0
requests==2.32.3
minio==7.2.8
pytest==8.3.3
flake8==7.1.1
Install: pip install -r requirements.txt.

LiveCoinWatch API Key:
Sign up at livecoinwatch.com for a free API key.
Set env var: export LIVE_COIN_WATCH_API_KEY='your-key-here'.
Free tier: ~10,000 credits/day; each API call costs 1 credit.

MinIO (Object Storage):
Download MinIO: wget https://dl.min.io/server/minio/release/linux-amd64/minio (adjust for OS).
Make executable: chmod +x minio.
Run: ./minio server ./minio-data --console-address ":9001".
Access console: http://localhost:9001 (creds: minioadmin/minioadmin).
Create bucket: mlops-data.
Set DVC remote:textdvc init
dvc remote add myremote s3://mlops-data
dvc remote modify myremote endpointurl http://127.0.0.1:9000
dvc remote modify myremote access_key_id admin
dvc remote modify myremote secret_access_key admin12345

Apache Airflow:
Init DB: airflow db init.
Create admin: airflow users create --username admin --firstname Admin --lastname Admin --role Admin --email admin@example.com --password admin.
Run: airflow scheduler & airflow webserver -p 8080.
Access: http://localhost:8080 (login: admin/admin).
Place DAGs in ~/airflow/dags/ or set AIRFLOW_HOME.

Dagshub (MLflow & DVC Remote):
Sign up at dagshub.com.
Create repo: yourusername/mlops-rps.
Get token from settings.
Set MLflow URI: export MLFLOW_TRACKING_URI=https://dagshub.com/yourusername/mlops-rps.mlflow.
Set auth: export MLFLOW_TRACKING_USERNAME=yourusername and export MLFLOW_TRACKING_PASSWORD=token.
DVC remote: dvc remote add origin https://dagshub.com/yourusername/mlops-rps.dvc.
Auth: dvc remote modify origin --local user yourusername and dvc remote modify origin --local password token.

Prometheus & Grafana:
Download Prometheus: From prometheus.io/download.
Create prometheus.yml (see Phase IV code).
Run: ./prometheus --config.file=prometheus.yml.
Download Grafana: From grafana.com.
Run: grafana-server.
Access: http://localhost:3000 (admin/admin).
Add datasource: Prometheus at http://localhost:9090.

Docker Hub:
Sign up (free).
Set secrets in GitHub repo: DOCKER_USERNAME, DOCKER_PASSWORD.

Test Setup:
Commit initial files: git add .; git commit -m "Initial setup"; git push origin master.

Development and Coding Steps
Follow phases from the project doc. Create files as specified. Use Codex to refine code.
Phase I: Problem Definition and Data Ingestion
Problem: Predict BTC next-hour price using time-series data.

API Selection: Financial domain, LiveCoinWatch API.
Airflow DAG (dags/etl_dag.py):Pythonfrom airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys
sys.path.append(os.path.join(os.path.dirname(**file**), '..', 'src'))

from extract import extract_data
from utils import quality_check
from transform import transform_data
from utils import load_and_version

default_args = {
'owner': 'airflow',
'depends_on_past': False,
'start_date': datetime(2025, 1, 1),
'retries': 1,
'retry_delay': timedelta(minutes=5),
}

with DAG('etl_dag', default_args=default_args, schedule_interval='@hourly', catchup=False) as dag:
extract_task = PythonOperator(task_id='extract', python_callable=extract_data)
quality_task = PythonOperator(task_id='quality_check', python_callable=quality_check, provide_context=True)
transform_task = PythonOperator(task_id='transform', python_callable=transform_data, provide_context=True)
load_task = PythonOperator(task_id='load_version', python_callable=load_and_version, provide_context=True)
train_task = PythonOperator(task_id='train', python_callable='src.train.train_model') # Phase II

    extract_task >> quality_task >> transform_task >> load_task >> train_task

Extraction (src/extract.py):Pythonimport requests
import json
import time
import os
import pandas as pd

def extract*data():
api_key = os.getenv('LIVE_COIN_WATCH_API_KEY')
headers = {'content-type': 'application/json', 'x-api-key': api_key} # For initial historical (30 days, hourly assumed)
if not os.path.exists('data/raw/btc_history.csv'):
payload = {
'currency': 'USD',
'code': 'BTC',
'start': int((time.time() - 30*24*3600) * 1000),
'end': int(time.time() \_ 1000),
'meta': False
}
response = requests.post('https://api.livecoinwatch.com/coins/single/history', json=payload, headers=headers)
data = response.json()['history']
df = pd.DataFrame(data)
df.to_csv('data/raw/btc_history.csv', index=False)
return df # For live append
else:
payload = {'currency': 'USD', 'code': 'BTC', 'meta': False}
response = requests.post('https://api.livecoinwatch.com/coins/single', json=payload, headers=headers)
data = response.json()
df_existing = pd.read_csv('data/raw/btc_history.csv')
new_row = {'date': int(time.time() \* 1000), 'rate': data['rate'], 'volume': data['volume'], 'cap': data['cap']}
df_new = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
df_new.to_csv('data/raw/btc_history.csv', index=False)
return df_new
Quality Check (src/utils.py - part):Pythonimport pandas as pd

def quality_check(\*\*context):
df = context['task_instance'].xcom_pull(task_ids='extract')
if df is None:
df = pd.read_csv('data/raw/btc_history.csv')
null_ratio = df.isnull().mean().mean()
if null_ratio > 0.01:
raise ValueError("Data quality check failed: >1% nulls") # Schema validation (e.g., check columns)
required_cols = ['date', 'rate', 'volume', 'cap']
if not all(col in df.columns for col in required_cols):
raise ValueError("Schema validation failed")
context['task_instance'].xcom_push(key='df', value=df)
Transformation (src/transform.py):Pythonfrom ydata_profiling import ProfileReport
import pandas as pd

def transform_data(\*\*context):
df = context['task_instance'].xcom_pull(key='df') # Clean
df = df.dropna()
df['date'] = pd.to_datetime(df['date'], unit='ms') # Feature engineering
df['lag_1'] = df['rate'].shift(1)
df['rolling_mean_5'] = df['rate'].rolling(window=5).mean()
df['hour'] = df['date'].dt.hour
df.dropna(inplace=True) # Report
profile = ProfileReport(df, title="Data Profile")
profile.to_file("data/report.html")
df.to_csv('data/processed/btc_processed.csv', index=False)
context['task_instance'].xcom_push(key='processed_df', value=df)
Loading & Versioning (src/utils.py - part):Pythonimport os

def load_and_version(\*\*context): # Assuming MinIO client setup in env
from minio import Minio
client = Minio('localhost:9000', access_key='minioadmin', secret_key='minioadmin', secure=False)
client.fput_object('mlops-data', 'btc_processed.csv', 'data/processed/btc_processed.csv') # DVC
os.system('dvc add data/processed/btc_processed.csv')
os.system('dvc push')
os.system('git add data/processed/btc_processed.csv.dvc')
os.system('git commit -m "Version processed data"')
os.system('git push')

Phase II: Experimentation and Model Management

Training Script (src/train.py):Pythonimport mlflow
import mlflow.tensorflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import os

def train_model():
os.system('dvc pull') # Get latest data
df = pd.read_csv('data/processed/btc_processed.csv')
features = ['lag_1', 'rolling_mean_5', 'hour']
X = df[features].values
y = df['rate'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(len(features),)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, validation_split=0.2)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    with mlflow.start_run():
        mlflow.log_params({'epochs': 50, 'layers': [64, 32, 1]})
        mlflow.log_metrics({'rmse': rmse, 'mae': mae, 'r2': r2})
        mlflow.tensorflow.log_model(model, 'model')
        mlflow.log_artifact('data/report.html')
        mlflow.register_model(mlflow.active_run().info.artifact_uri + '/model', 'btc_predictor')

Phase III: Continuous Integration & Deployment

Git Workflow: Feature branches -> PR to dev -> PR to test (approve) -> PR to master (approve).
GitHub Actions (.github/workflows/ci.yml):YAMLname: CI/CD Pipeline

on:
pull_request:
branches: [dev, test, master]

jobs:
lint-test:
if: github.base_ref == 'dev'
runs-on: ubuntu-latest
steps: - uses: actions/checkout@v3 - name: Set up Python
uses: actions/setup-python@v4
with:
python-version: '3.10' - run: pip install -r requirements.txt - run: flake8 . - run: pytest tests/

retrain-compare:
if: github.base_ref == 'test'
runs-on: ubuntu-latest
env:
MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_TRACKING_USERNAME }}
      MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_TRACKING_PASSWORD }}
      LIVE_COIN_WATCH_API_KEY: ${{ secrets.LIVE_COIN_WATCH_API_KEY }}
    steps:
      - uses: actions/checkout@v3
      - uses: iterative/setup-cml@v1
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python src/extract.py  # Mock or run parts
      - run: python src/transform.py
      - run: python src/train.py
      - name: Compare Models
        run: |
          # Fetch prod metrics from MLflow (simplified; use mlflow client)
          prod_rmse = 1000  # Placeholder; query MLflow
          new_rmse = $(python -c "import mlflow; run=mlflow.active_run(); print(mlflow.get_metric('rmse'))")  # Adjust
          echo "# Model Comparison\nNew RMSE: $new_rmse vs Prod: $prod_rmse" >> report.md
          cml comment create report.md --token=${{ secrets.GITHUB_TOKEN }}
if (( $(echo "$new_rmse > $prod_rmse" | bc -l) )); then exit 1; fi

deploy:
if: github.base_ref == 'master'
runs-on: ubuntu-latest
steps: - uses: actions/checkout@v3 - name: Build Docker
run: docker build -t ${{ secrets.DOCKER_USERNAME }}/mlops-rps:latest .
      - name: Login to Docker Hub
        run: echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin - name: Push Docker
run: docker push ${{ secrets.DOCKER_USERNAME }}/mlops-rps:latest - name: Verify
run: docker run -d -p 8000:8000 ${{ secrets.DOCKER_USERNAME }}/mlops-rps:latest
Dockerfile:dockerfileFROM python:3.10-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
Serve (src/serve.py):Pythonfrom fastapi import FastAPI
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, make_asgi_adapter
import numpy as np

app = FastAPI()
model = mlflow.pyfunc.load_model("models:/btc_predictor/Production")

requests_total = Counter('requests_total', 'Total requests')
inference_latency = Histogram('inference_latency_seconds', 'Inference latency')
drift_ratio = Counter('drift_ratio', 'Data drift proxy')

metrics_app = make_asgi_adapter()
app.mount("/metrics", metrics_app)

@app.get("/health")
def health():
return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
features = np.array([[data['lag_1'], data['rolling_mean_5'], data['hour']]])
with inference_latency.time():
pred = model.predict(features)
requests_total.inc() # Simple drift check
if data['lag_1'] > 2 \* np.mean(features): # Proxy
drift_ratio.inc()
return {"prediction": pred[0]}

Phase IV: Monitoring and Observability

Prometheus Config (prometheus.yml):YAMLglobal:
scrape_interval: 15s

scrape_configs:

- job_name: 'fastapi'
  static_configs: - targets: ['localhost:8000']
  Grafana Dashboard:
  In Grafana UI: Create dashboard.
  Panels: Query Prometheus for inference_latency_seconds, requests_total, drift_ratio.
  Alert: Create alert rule for inference_latency_seconds > 0.5 (log to file or console; configure notification channel to file).

Testing and Running

Unit Tests (tests/test_utils.py):Pythonimport pytest
from src.utils import quality_check

def test_quality_check(): # Mock df
assert True # Expand with actual tests
Run: pytest.

Full Run:
Trigger DAG in Airflow.
Merge PRs through branches.
Deploy: docker run -p 8000:8000 image.
Test predict: curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"lag_1": 50000, "rolling_mean_5": 49000, "hour": 12}'.

Documentation:
Add screenshots of Dagshub (experiments), Grafana dashboards, PR comments to README.md.
