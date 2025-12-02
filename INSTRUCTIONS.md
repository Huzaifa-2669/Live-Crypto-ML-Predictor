## MinIO Setup & Test Upload

- MinIO is defined in `Live-Crypto-ML-Predictor/docker-compose.minio.yml` and runs:

  - S3 API on `http://localhost:9000`
  - Web console on `http://localhost:9001`

- Configure env in `Live-Crypto-ML-Predictor/.env.minio`:

  - `MINIO_ENDPOINT=http://localhost:9000`
  - `MINIO_ACCESS_KEY=admin`
  - `MINIO_SECRET_KEY=admin12345`
  - `MINIO_BUCKET=mlops-data`

- Start MinIO:

  - `docker compose -f Live-Crypto-ML-Predictor/docker-compose.minio.yml up -d`

- Test upload script:
  - `python Live-Crypto-ML-Predictor/scripts/minio_test_upload.py`
  - Creates bucket if missing, uploads `samples/hello.txt`, and lists objects.

# MLOps Case Study: Real-Time Predictive System (RPS) - Project Guide

**Role:** MLOps Lead Architect
**Date:** December 2, 2024
**Target Audience:** Junior MLOps Engineer (Student)

## 1. Project Overview & Strategy

You are building a system to predict **Bitcoin (BTC) Price/Volatility** for the next hour.

**The Core Challenge:** The provided API gives the _current_ state. Standard ML requires _historical_ data.
**The Solution:** Your Airflow DAG will act as a "Heartbeat." Every time it runs (e.g., every hour), it fetches the current price and _appends_ it to a dataset. This creates the history needed for training.

**Prerequisites:**

1.  **GitHub Account** (Repository created).
2.  **Dagshub Account** (Connect to your GitHub Repo - this gives you free MLflow & DVC remote storage).
3.  **Docker Desktop** (Installed and running).
4.  **Python 3.9+**.
5.  **Live Coin Watch API Key**.

---

## 2. Phase I: Data Ingestion & Orchestration (Airflow)

We need to set up Airflow to run a pipeline that fetches data, cleans it, and versions it.

### Step 2.1: Project Structure

Create this folder structure. It is industry standard.

```text
my-mlops-project/
├── .github/workflows/   # CI/CD
├── airflow/
│   ├── dags/            # The Orchestrator
│   └── docker-compose.yaml
├── data/                # Local data (ignored by git)
├── src/                 # Source code
│   ├── extraction.py
│   ├── processing.py
│   ├── train.py
│   └── inference.py
├── tests/               # Unit tests
├── Dockerfile           # For the API serving
├── requirements.txt
└── .dvc/                # DVC configuration
```

Dags hub access token : 923be5edf31679dc179fe9349e01d2e54e78cc10
