"""FastAPI serving app with Prometheus metrics."""
import os
from fastapi import FastAPI
import mlflow.pyfunc
import numpy as np
from prometheus_client import Counter, Histogram, make_asgi_app

app = FastAPI()

# Use env-driven model URI; default to a registry stage if not provided.
DEFAULT_MODEL_URI = "models:/btc_predictor/Production"
model_uri = os.environ.get("MODEL_URI", DEFAULT_MODEL_URI)
model = mlflow.pyfunc.load_model(model_uri)

requests_total = Counter("requests_total", "Total requests")
inference_latency = Histogram("inference_latency_seconds", "Inference latency")
drift_ratio = Counter("drift_ratio", "Data drift proxy")

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": model_uri}


@app.get("/")
def root():
    return {
        "message": "BTC predictor service",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
        },
        "model_uri": model_uri,
    }


@app.post("/predict")
def predict(payload: dict):
    features = np.array([[payload["lag_1"], payload["rolling_mean_5"], payload["hour"]]])
    with inference_latency.time():
        pred = model.predict(features)
    requests_total.inc()
    if payload["lag_1"] > 2 * np.mean(features):
        drift_ratio.inc()
    return {"prediction": float(pred[0])}
