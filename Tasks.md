# To Do
- Stabilize Airflow run (etl_dag): align AIRFLOW_HOME/log settings and env vars; adjust DAG import timeout if needed; rerun end-to-end.
- Add Prometheus/Grafana monitoring for serving metrics.
- CI/CD setup (dev/test/master + GitHub Actions/CML) once pipeline is stable; add lint/tests.
- Consider lighter serving requirements if Docker image size is a concern.

# Done
- Read and summarized project instructions (Project_Instructions.md, INSTRUCTIONS.md, MLOps Project Spring 25.pdf).
- Updated `.env.minio` to bucket `mlops-data`; started MinIO via docker-compose; test upload verified (samples/hello.txt in bucket).
- DVC initialized with local cache dir; remote `minio` configured (s3://mlops-data @ http://localhost:9000); processed data tracked and pushed; also uploaded processed data/report to DagsHub bucket.
- Scaffolded project structure and ETL/train code; added logging and runnable entry points; README added; Dockerfile for serving added.
- MLflow pointed to DagsHub, pinned to mlflow 2.13.1; model registered as btc_predictor v1; serving updated to env-driven MODEL_URI with root endpoint.
