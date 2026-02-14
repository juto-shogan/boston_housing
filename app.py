from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.predict import FEATURE_COLUMNS, predict_price

MODEL_PATH = Path("models/boston_housing_model.joblib")
METRICS_PATH = Path("models/metrics.json")

DEFAULT_VALUES = {
    "crim": 0.25,
    "zn": 0.0,
    "indus": 11.0,
    "chas": 0,
    "nox": 0.55,
    "rm": 6.2,
    "age": 70.0,
    "dis": 4.0,
    "rad": 4,
    "tax": 300,
    "ptratio": 18.0,
    "b": 390.0,
    "lstat": 12.0,
}

st.set_page_config(page_title="Boston Housing Prediction", page_icon="🏠", layout="wide")

st.title("Boston Housing Price Prediction")
st.caption("Predict median home value (medv) from Boston housing features.")

if not MODEL_PATH.exists():
    st.warning(
        "No trained model found yet. Run `python src/train.py` first, then refresh this app."
    )

metrics = None
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())

with st.sidebar:
    st.header("Model input features")
    user_input = {
        "crim": st.number_input("CRIM", min_value=0.0, value=DEFAULT_VALUES["crim"], step=0.01),
        "zn": st.number_input("ZN", min_value=0.0, value=DEFAULT_VALUES["zn"], step=1.0),
        "indus": st.number_input("INDUS", min_value=0.0, value=DEFAULT_VALUES["indus"], step=0.1),
        "chas": st.selectbox("CHAS", options=[0, 1], index=DEFAULT_VALUES["chas"]),
        "nox": st.number_input("NOX", min_value=0.0, max_value=1.0, value=DEFAULT_VALUES["nox"], step=0.001),
        "rm": st.number_input("RM", min_value=0.0, value=DEFAULT_VALUES["rm"], step=0.01),
        "age": st.number_input("AGE", min_value=0.0, max_value=100.0, value=DEFAULT_VALUES["age"], step=1.0),
        "dis": st.number_input("DIS", min_value=0.0, value=DEFAULT_VALUES["dis"], step=0.01),
        "rad": st.number_input("RAD", min_value=1, value=DEFAULT_VALUES["rad"], step=1),
        "tax": st.number_input("TAX", min_value=0, value=DEFAULT_VALUES["tax"], step=1),
        "ptratio": st.number_input("PTRATIO", min_value=0.0, value=DEFAULT_VALUES["ptratio"], step=0.1),
        "b": st.number_input("B", min_value=0.0, value=DEFAULT_VALUES["b"], step=0.1),
        "lstat": st.number_input("LSTAT", min_value=0.0, value=DEFAULT_VALUES["lstat"], step=0.01),
    }

st.subheader("Prediction")
if st.button("Predict median value", type="primary", disabled=not MODEL_PATH.exists()):
    ordered_input = {feature: float(user_input[feature]) for feature in FEATURE_COLUMNS}
    prediction = predict_price(ordered_input)
    st.success(f"Predicted medv: **{prediction:.2f}**")

if metrics:
    st.subheader("Last training metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['mae']:.3f}")
    c2.metric("RMSE", f"{metrics['rmse']:.3f}")
    c3.metric("R²", f"{metrics['r2']:.3f}")
