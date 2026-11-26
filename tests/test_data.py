import pandas as pd

from src.data import add_booking_date, chronological_split
from src.model import EXCLUDED, split_features


def _toy():
    return pd.DataFrame({
        "arrival_date_year": [2016, 2016, 2017, 2017],
        "arrival_date_month": ["July", "July", "August", "August"],
        "arrival_date_day_of_month": [1, 10, 5, 20],
        "lead_time": [0, 9, 3, 100],
        "is_canceled": [0, 1, 0, 1],
        "reservation_status": ["Check-Out", "Canceled", "Check-Out", "Canceled"],
        "reservation_status_date": ["2016-07-01"] * 4,
        "booking_changes": [0, 0, 0, 0],
        "assigned_room_type": ["A"] * 4,
        "company": [None] * 4,
        "adr": [10.0, 20.0, 30.0, 40.0],
    })


def test_booking_date_is_arrival_minus_lead_time():
    df = add_booking_date(_toy())
    assert str(df.loc[0, "booking_date"].date()) == "2016-07-01"
    assert str(df.loc[1, "booking_date"].date()) == "2016-07-01"


def test_split_is_chronological_and_complete():
    train, val, test = chronological_split(_toy(), train_frac=0.5, val_frac=0.25)
    assert len(train) + len(val) + len(test) == 4
    assert train["booking_date"].max() <= test["booking_date"].min()


def test_leakage_and_unused_columns_are_removed():
    X, y = split_features(_toy())
    for col in EXCLUDED:
        assert col not in X.columns
    assert list(y) == [0, 1, 0, 1]
