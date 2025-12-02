"""FastAPI serving app with Prometheus metrics."""
from fastapi import FastAPI
import mlflow.pyfunc
import numpy as np
from prometheus_client import Counter, Histogram, make_asgi_app

app = FastAPI()
model = mlflow.pyfunc.load_model("models:/btc_predictor/Production")

requests_total = Counter("requests_total", "Total requests")
inference_latency = Histogram("inference_latency_seconds", "Inference latency")
drift_ratio = Counter("drift_ratio", "Data drift proxy")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: dict):
    features = np.array([[payload["lag_1"], payload["rolling_mean_5"], payload["hour"]]])
    with inference_latency.time():
        pred = model.predict(features)
    requests_total.inc()
    if payload["lag_1"] > 2 * np.mean(features):
        drift_ratio.inc()
    return {"prediction": float(pred[0])}
