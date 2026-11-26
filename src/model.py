"""Feature preparation and the classification pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

# Columns never used as features: the target, the date helper, and leakage/unused columns.
EXCLUDED = [config.TARGET, "booking_date", *config.LEAKAGE_COLS, *config.DROP_COLS]


def split_features(df: pd.DataFrame):
    """Return (X, y) with the excluded columns removed."""
    X = df.drop(columns=[c for c in EXCLUDED if c in df.columns])
    return X, df[config.TARGET]


def build_pipeline(n_estimators=300, max_depth=12, min_samples_leaf=3) -> Pipeline:
    """Preprocess and classify. Column types are picked from the data, so the same
    pipeline works for a full training frame and for one API request."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                make_column_selector(dtype_include="number"),
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                make_column_selector(dtype_exclude="number"),
            ),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )),
    ])
