# To Do
- Run Airflow DAG end-to-end (extract → quality → transform → load/version → train) inside Airflow.
- Containerize serving (FastAPI) with env MODEL_URI (e.g., models:/btc_predictor/1 or Production); expose /health, /predict, /metrics.
- Add Prometheus/Grafana monitoring for latency/request/drift metrics.
- CI/CD setup (dev/test/master + GitHub Actions/CML) once pipeline is stable; add lint/tests.

# Done
- Read and summarized project instructions (Project_Instructions.md, INSTRUCTIONS.md, MLOps Project Spring 25.pdf).
- Updated `.env.minio` to bucket `mlops-data`; started MinIO via docker-compose; test upload verified (samples/hello.txt in bucket).
- DVC initialized with local cache dir; remote `minio` configured (s3://mlops-data @ http://localhost:9000); processed data tracked and pushed; also uploaded processed data/report to DagsHub bucket.
- Scaffolded project structure and ETL/train code; added logging and runnable entry points.
- MLflow pointed to DagsHub, pinned to mlflow 2.13.1; model registered as btc_predictor v1; serving updated to env-driven MODEL_URI with root endpoint.
