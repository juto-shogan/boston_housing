from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/data.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "boston_housing_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
TARGET_COLUMN = "medv"


def train_model(data_path: Path = DATA_PATH) -> dict[str, float]:
    df = pd.read_csv(data_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    model_metrics = train_model()
    print("Training complete")
    print(json.dumps(model_metrics, indent=2))
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")
