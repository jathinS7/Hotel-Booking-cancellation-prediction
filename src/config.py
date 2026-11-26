"""Paths and constants shared by the training script and the API."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "hotel_bookings.csv"
MODEL_PATH = BASE_DIR / "models" / "hotel_cancellation_model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "decision_threshold.json"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"

TARGET = "is_canceled"
RANDOM_STATE = 42

# Columns that are unknown at booking time or leak the outcome.
# reservation_status / reservation_status_date are set when the booking ends.
LEAKAGE_COLS = ["reservation_status", "reservation_status_date"]
# Dropped for the same reasons as the notebook: uninformative or too sparse.
DROP_COLS = ["assigned_room_type", "booking_changes", "company"]
