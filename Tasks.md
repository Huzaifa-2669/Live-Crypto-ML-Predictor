# To Do
- Point MLflow to Dagshub (MLFLOW_TRACKING_URI/USERNAME/PASSWORD) and rerun training to log/register model there.
- Wire Airflow to run the DAG (extract → quality → transform → load/version → train) and validate end-to-end.
- Add serving/monitoring wiring once a registered model exists (FastAPI + Prometheus metrics).
- CI/CD setup (dev/test/master branching, GitHub Actions + CML) after pipeline is stable.

# Done
- Read and summarized project instructions (Project_Instructions.md, INSTRUCTIONS.md, MLOps Project Spring 25.pdf).
- Updated `.env.minio` to bucket `mlops-data`; started MinIO via docker-compose; test upload verified (samples/hello.txt in bucket).
- DVC initialized with local cache dir; remote `minio` configured (s3://mlops-data @ http://localhost:9000); processed data tracked and pushed.
- Scaffolded project structure and ETL/train code; added logging and runnable entry points.
- Ran extract (data/raw/btc_history.csv), transform (data/processed/btc_processed.csv, data/report.html), and training (local MLflow) successfully.
