# Boston Housing Price Prediction

A machine learning project to train and serve a Boston Housing regression model using Streamlit.

## What is implemented

- Reproducible training pipeline (`src/train.py`) that:
  - reads `data/data.csv`
  - trains a `RandomForestRegressor`
  - evaluates on a holdout test set (MAE / RMSE / R²)
  - saves artifacts to `models/`
- Prediction helper (`src/predict.py`) for consistent feature ordering and model loading.
- Streamlit UI (`app.py`) with:
  - form controls for all model features
  - one-click prediction
  - display of last recorded training metrics

## Project structure

```text
.
├── app.py
├── data/
│   ├── data.csv
│   └── details.png
├── models/                # generated after training
├── src/
│   ├── predict.py
│   └── train.py
├── sample.ipynb
└── requirements.txt
```

## Quick start

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python src/train.py
```

4. Run the app:

```bash
streamlit run app.py
```

## Notes

- This dataset is commonly used for education/demo purposes; it also has known ethical limitations in modern ML fairness contexts.
- For production use, add stronger validation, model versioning, CI checks, and monitoring.
