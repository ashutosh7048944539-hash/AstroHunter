from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_FILE = Path("data/ml/extracted_features.csv")


FEATURE_COLUMNS = [
    "period_days",
    "transit_depth",
    "duration_hours",
    "num_transit_points",
    "baseline_scatter",
    "transit_snr",
    "num_events",
    "period_scatter_days",
]


def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def analyze_missingness(df: pd.DataFrame) -> None:
    print_section("Missingness by Class")

    rows = []

    for label in [0, 1]:

        subset = df[df["label"] == label]

        for feature in FEATURE_COLUMNS:

            missing = subset[feature].isna().sum()

            rows.append(
                {
                    "label": label,
                    "feature": feature,
                    "missing": missing,
                    "total": len(subset),
                    "missing_pct": (
                        100.0 * missing / len(subset)
                    ),
                }
            )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "missing_pct": "{:.1f}".format
            },
        )
    )


def analyze_statistics(df: pd.DataFrame) -> None:
    print_section("Overall Feature Statistics")

    print(
        df[FEATURE_COLUMNS]
        .describe()
        .T
        .to_string()
    )


def analyze_by_class(df: pd.DataFrame) -> None:
    print_section("Feature Statistics by Class")

    for label in [0, 1]:

        subset = df[
            df["label"] == label
        ]

        print()
        print(
            f"Label {label} "
            f"({'negative' if label == 0 else 'positive'})"
        )

        print(
            subset[FEATURE_COLUMNS]
            .describe()
            .T[
                [
                    "count",
                    "mean",
                    "50%",
                    "std",
                    "min",
                    "max",
                ]
            ]
            .to_string()
        )


def analyze_period_availability(
    df: pd.DataFrame,
) -> None:

    print_section(
        "Period Availability by Class"
    )

    for label in [0, 1]:

        subset = df[
            df["label"] == label
        ]

        known_period = (
            subset["period_days"]
            .notna()
        )

        print(
            f"Label {label}:"
        )

        print(
            f"  Period measurable: "
            f"{known_period.sum()} / "
            f"{len(subset)}"
        )

        print(
            f"  Period missing:     "
            f"{(~known_period).sum()} / "
            f"{len(subset)}"
        )


def analyze_extraction_status(
    df: pd.DataFrame,
) -> None:

    print_section(
        "Extraction Status by Class"
    )

    table = pd.crosstab(
        df["label"],
        df["extraction_status"],
    )

    print(
        table.to_string()
    )


def analyze_event_categories(
    df: pd.DataFrame,
) -> None:

    print_section(
        "Event / Periodicity Categories"
    )

    # Category 1:
    # No detected candidate event.
    no_events = (
        df["extraction_status"]
        == "no_candidate_events"
    )

    # Category 2:
    # Candidate event(s), but no reliable period.
    candidate_no_period = (
        (~no_events)
        & df["period_days"].isna()
    )

    # Category 3:
    # Reliable period estimate.
    reliable_period = (
        df["period_days"].notna()
    )

    categories = pd.DataFrame(
        {
            "no_candidate_events": no_events,
            "candidate_but_no_period":
                candidate_no_period,
            "reliable_period":
                reliable_period,
        }
    )

    for label in [0, 1]:

        subset = categories[
            df["label"] == label
        ]

        print()
        print(
            f"Label {label}"
        )

        for column in categories.columns:

            count = subset[column].sum()

            print(
                f"  {column:30s}: "
                f"{count:2d}"
            )


def analyze_suspicious_values(
    df: pd.DataFrame,
) -> None:

    print_section(
        "Potentially Suspicious Values"
    )

    # Extremely short inferred periods.
    short_period = (
        df["period_days"].notna()
        & (df["period_days"] < 0.5)
    )

    print(
        f"Periods < 0.5 days: "
        f"{short_period.sum()}"
    )

    if short_period.any():

        print(
            df.loc[
                short_period,
                [
                    "tic_id",
                    "label",
                    "period_days",
                    "duration_hours",
                    "num_events",
                    "period_scatter_days",
                ],
            ].to_string(
                index=False
            )
        )

    # Very long durations.
    long_duration = (
        df["duration_hours"].notna()
        & (df["duration_hours"] > 10)
    )

    print()
    print(
        f"Durations > 10 hours: "
        f"{long_duration.sum()}"
    )

    if long_duration.any():

        print(
            df.loc[
                long_duration,
                [
                    "tic_id",
                    "label",
                    "period_days",
                    "duration_hours",
                    "transit_depth",
                    "transit_snr",
                ],
            ].to_string(
                index=False
            )
        )

    # Large period scatter.
    large_scatter = (
        df["period_scatter_days"].notna()
        & (
            df["period_scatter_days"]
            > 0.1
        )
    )

    print()
    print(
        f"Period scatter > 0.1 days: "
        f"{large_scatter.sum()}"
    )

    if large_scatter.any():

        print(
            df.loc[
                large_scatter,
                [
                    "tic_id",
                    "label",
                    "period_days",
                    "num_events",
                    "period_scatter_days",
                ],
            ].to_string(
                index=False
            )
        )


def analyze_period_consistency(
    df: pd.DataFrame,
) -> None:

    print_section(
        "Period Consistency"
    )

    subset = df[
        df["period_days"].notna()
        & df["period_scatter_days"].notna()
        & (df["period_days"] > 0)
    ].copy()

    if subset.empty:

        print(
            "No targets have measurable periods."
        )

        return

    subset["relative_period_scatter"] = (
        subset["period_scatter_days"]
        / subset["period_days"]
    )

    print(
        subset[
            [
                "tic_id",
                "label",
                "period_days",
                "num_events",
                "period_scatter_days",
                "relative_period_scatter",
            ]
        ]
        .sort_values(
            "relative_period_scatter"
        )
        .to_string(
            index=False
        )
    )


def main() -> None:

    print(
        "=== Day 4: Feature Quality Analysis ==="
    )

    if not FEATURE_FILE.exists():

        print(
            f"ERROR: Feature file not found:"
        )

        print(
            f"       {FEATURE_FILE}"
        )

        return

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"Loaded targets: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    analyze_extraction_status(df)

    analyze_missingness(df)

    analyze_period_availability(df)

    analyze_statistics(df)

    analyze_by_class(df)

    analyze_event_categories(df)

    analyze_suspicious_values(df)

    analyze_period_consistency(df)

    print_section(
        "Analysis Complete"
    )


if __name__ == "__main__":
    main()