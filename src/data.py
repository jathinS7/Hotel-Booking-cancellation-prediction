"""Loading and chronological splitting of the hotel bookings data."""

import pandas as pd

from src import config


def load_bookings(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and drop exact duplicate rows."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download hotel_bookings.csv and place it in data/."
        )
    return pd.read_csv(path).drop_duplicates().reset_index(drop=True)


def add_booking_date(df: pd.DataFrame) -> pd.DataFrame:
    """Booking date = arrival date minus lead time. Used only to order the rows."""
    df = df.copy()
    arrival = pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-"
        + df["arrival_date_month"]
        + "-"
        + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )
    df["booking_date"] = arrival - pd.to_timedelta(df["lead_time"], unit="D")
    return df


def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    """Split by booking date so the test set is the most recent bookings."""
    df = add_booking_date(df).sort_values("booking_date").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
