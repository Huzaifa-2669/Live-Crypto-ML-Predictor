# To Do
- Verify remaining core deps in venv (requests, pandas, dvc[s3]) and install if missing.
- Create project scaffold (src, data, airflow/dags, tests) aligned with instructions.
- Set up Airflow DAG and basic ETL/training scripts (after scaffold).
- Integrate MLflow/Dagshub configs and model registry usage.
- Add monitoring endpoints (Prometheus) and serve app (FastAPI) once model exists.
- CI/CD setup (dev/test/master branching, GitHub Actions + CML) after core pipeline is stable.

# Done
- Read and summarized project instructions (Project_Instructions.md, INSTRUCTIONS.md, MLOps Project Spring 25.pdf).
- Updated `.env.minio` to bucket `mlops-data`; started MinIO via docker-compose; test upload verified (samples/hello.txt in bucket).
- DVC initialized with local cache dir; remote `minio` configured (s3://mlops-data @ http://localhost:9000).
