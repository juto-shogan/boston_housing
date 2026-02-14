from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/boston_housing_model.joblib")

FEATURE_COLUMNS = [
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "b",
    "lstat",
]


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict_price(features: dict[str, float], model_path: Path = MODEL_PATH) -> float:
    model = load_model(model_path)
    ordered = {column: features[column] for column in FEATURE_COLUMNS}
    frame = pd.DataFrame([ordered])
    prediction = model.predict(frame)
    return float(prediction[0])
