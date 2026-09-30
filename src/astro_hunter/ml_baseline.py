from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path("data/ml/extracted_features.csv")

FEATURES = [
    "period_days",
    "transit_depth",
    "duration_hours",
    "num_transit_points",
    "baseline_scatter",
    "transit_snr",
    "num_events",
    "period_scatter_days",
]

TARGET = "label"


def main():
    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    print("=== Day 5: Logistic Regression Baseline ===")
    print(f"Total samples: {len(df)}")
    print(f"Number of features: {len(FEATURES)}")
    print(f"Positive samples: {(y == 1).sum()}")
    print(f"Negative samples: {(y == 0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42)),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Dataset Split ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    print("\n=== Metrics ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    classifier = model.named_steps["classifier"]
    coefficients = pd.Series(
        classifier.coef_[0],
        index=FEATURES,
    ).sort_values(ascending=False)

    print("\n=== Logistic Regression Coefficients ===")

    for feature, coefficient in coefficients.items():
        print(f"{feature:25s}: {coefficient:+.4f}")


if __name__ == "__main__":
    main()