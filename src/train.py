"""Training step: simple baseline model with MLflow logging."""
import logging
import os
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)


def train_model() -> None:
    # Pull latest data via DVC
    os.system("dvc pull")
    df = pd.read_csv("data/processed/btc_processed.csv")
    logger.info("Loaded processed data with shape %s", df.shape)
    features = ["lag_1", "rolling_mean_5", "hour"]
    X = df[features].values
    y = df["rate"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    # `squared` may not be supported in older sklearn; compute RMSE manually
    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    with mlflow.start_run():
        mlflow.log_params({"model": "RandomForestRegressor", "n_estimators": 100})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.sklearn.log_model(model, "model")
        if os.path.exists("data/report.html"):
            mlflow.log_artifact("data/report.html")
    logger.info("Training complete: rmse=%.4f mae=%.4f r2=%.4f", rmse, mae, r2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    train_model()
