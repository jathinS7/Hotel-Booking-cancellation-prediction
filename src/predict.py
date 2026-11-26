"""Load the saved model once and score bookings."""

import json
from functools import lru_cache

import joblib
import pandas as pd

from src import config


@lru_cache(maxsize=1)
def load_model():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {config.MODEL_PATH}. Run `python -m src.train` first."
        )
    model = joblib.load(config.MODEL_PATH)
    threshold = json.loads(config.THRESHOLD_PATH.read_text())["threshold"]
    return model, threshold


def predict_booking(booking: dict) -> dict:
    model, threshold = load_model()
    row = pd.DataFrame([booking])
    proba = float(model.predict_proba(row)[:, 1][0])
    is_cancel = proba >= threshold
    return {
        "cancellation_probability": round(proba, 4),
        "threshold": round(threshold, 4),
        "prediction": int(is_cancel),
        "action_recommended": (
            "Contact the guest to confirm the booking"
            if is_cancel
            else "No action required"
        ),
    }
