from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
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

    importance_results = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)

        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="roc_auc",
            n_repeats=20,
            random_state=42,
        )

        fold_importance = pd.DataFrame(
            {
                "feature": FEATURES,
                "importance": result.importances_mean,
                "std": result.importances_std,
                "fold": fold,
            }
        )

        importance_results.append(fold_importance)

    results = pd.concat(
        importance_results,
        ignore_index=True,
    )

    summary = (
        results
        .groupby("feature")
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
        )
        .sort_values(
            "mean_importance",
            ascending=False,
        )
    )

    print("=== Permutation Importance ===")

    for feature, row in summary.iterrows():
        print(
            f"{feature:25s}: "
            f"{row['mean_importance']:+.4f} "
            f"+/- {row['std_importance']:.4f}"
        )


if __name__ == "__main__":
    main()