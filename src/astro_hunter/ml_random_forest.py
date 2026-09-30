from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


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

    print("=== Day 5: Random Forest Baseline ===")
    print(f"Total samples: {len(df)}")
    print(f"Number of features: {len(FEATURES)}")
    print(f"Positive samples: {(y == 1).sum()}")
    print(f"Negative samples: {(y == 0).sum()}")

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
    )

    print("\n=== 5-Fold ROC-AUC ===")
    print("Scores:", scores.round(4))
    print(f"Mean: {scores.mean():.4f}")
    print(f"Std:  {scores.std():.4f}")

    # Train once on the full dataset so we can inspect
    # feature importance.
    model.fit(X, y)

    classifier = model.named_steps["classifier"]

    importances = pd.Series(
        classifier.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)

    print("\n=== Random Forest Feature Importance ===")

    for feature, importance in importances.items():
        print(f"{feature:25s}: {importance:.4f}")


if __name__ == "__main__":
    main()