"""Train the cancellation classifier and save the model, threshold and metrics.

Method (same as the notebook):
1. Chronological split by booking date: 70% train, 15% validation, 15% test.
2. Fit on train, pick the F1-optimal decision threshold on validation.
3. Report test metrics with that threshold.
4. Refit on train + validation and save that model for the API.
"""

import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from src import config
from src.data import chronological_split, load_bookings
from src.model import build_pipeline, split_features


def best_f1_threshold(y_true, proba) -> float:
    """Threshold that maximises F1 on the given probabilities."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
    return float(thresholds[np.argmax(f1)])


def evaluate(name, y_true, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    metrics = {
        "model": name,
        "threshold": round(threshold, 4),
        "accuracy": accuracy(y_true, pred),
        "precision": round(float((pred[y_true == 1] == 1).sum() / max((pred == 1).sum(), 1)), 4),
        "recall": round(float((pred[y_true == 1] == 1).sum() / max((y_true == 1).sum(), 1)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
    }
    print(f"\n{name} (threshold={threshold:.4f})")
    print(confusion_matrix(y_true, pred))
    print(classification_report(y_true, pred, digits=3))
    return metrics


def accuracy(y_true, pred) -> float:
    return round(float((np.asarray(y_true) == pred).mean()), 4)


def main():
    df = load_bookings()
    train_df, val_df, test_df = chronological_split(df)

    X_train, y_train = split_features(train_df)
    X_val, y_val = split_features(val_df)
    X_test, y_test = split_features(test_df)

    model = build_pipeline()
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = best_f1_threshold(y_val, val_proba)
    print(f"Validation-chosen threshold: {threshold:.4f}")

    test_proba = model.predict_proba(X_test)[:, 1]
    results = [evaluate("Trained on train only, test set", y_test, test_proba, threshold)]

    # Final model: refit on train + validation with the same settings.
    final_model = clone(model)
    final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    final_proba = final_model.predict_proba(X_test)[:, 1]
    results.append(evaluate("Final model (train + validation), test set", y_test, final_proba, threshold))

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import joblib  # imported here so the rest of the package works without it

    joblib.dump(final_model, config.MODEL_PATH)
    config.THRESHOLD_PATH.write_text(json.dumps({"threshold": threshold}, indent=2))
    config.METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
