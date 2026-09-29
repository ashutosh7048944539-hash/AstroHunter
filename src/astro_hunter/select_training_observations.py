from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "tess_available_targets.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "selected_training_observations.csv"
)

# Number of independent TIC targets per class.
TARGETS_PER_CLASS = 50

# Reproducible sampling.
RANDOM_STATE = 42


# ============================================================
# LOAD
# ============================================================

def load_manifest():
    """Load the MAST availability manifest."""

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"File not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = {
        "toi",
        "tid",
        "label",
        "tfopwg_disp",
        "obs_id",
        "obsid",
        "sequence_number",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


# ============================================================
# CLEAN
# ============================================================

def clean_manifest(dataframe):
    """
    Standardize identifiers and remove invalid observations.
    """

    df = dataframe.copy()

    # TIC IDs must be integers.
    df["tid"] = pd.to_numeric(
        df["tid"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["tid"]
    )

    df["tid"] = (
        df["tid"]
        .astype("int64")
    )

    # Labels must be 0/1.
    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["label"]
    )

    df["label"] = (
        df["label"]
        .astype(int)
    )

    # Sector number.
    df["sequence_number"] = pd.to_numeric(
        df["sequence_number"],
        errors="coerce",
    )

    # Remove rows without an observation ID.
    df["obs_id"] = (
        df["obs_id"]
        .astype(str)
        .str.strip()
    )

    df = df[
        (df["obs_id"] != "")
        & (df["obs_id"] != "nan")
    ].copy()

    return df


# ============================================================
# ONE OBSERVATION PER TIC
# ============================================================

def select_one_observation_per_tic(
    dataframe,
):
    """
    Select exactly one TESS/SPOC observation per TIC.

    This ensures that each astronomical target contributes
    at most one light curve to the first ML dataset.

    We prefer:
        1. 120-second cadence
        2. otherwise shortest exposure time
        3. deterministic ordering
    """

    df = dataframe.copy()

    # --------------------------------------------------------
    # MAST exposure/cadence metadata.
    #
    # t_exptime is in seconds.
    # 120 seconds corresponds to the preferred 2-minute
    # cadence product.
    # --------------------------------------------------------

    if "t_exptime" in df.columns:

        df["t_exptime"] = pd.to_numeric(
            df["t_exptime"],
            errors="coerce",
        )

        # Distance from 120 seconds.
        df["cadence_distance"] = (
            (df["t_exptime"] - 120)
            .abs()
        )

    else:

        df["cadence_distance"] = 0.0

    # Prefer shorter exposure/cadence after 120-sec preference.
    df["exposure_sort"] = (
        df["t_exptime"]
        if "t_exptime" in df.columns
        else 0
    )

    # Deterministic sorting.
    df = df.sort_values(
        by=[
            "tid",
            "cadence_distance",
            "exposure_sort",
            "sequence_number",
            "obs_id",
        ],
        na_position="last",
    )

    # Keep exactly one observation per TIC.
    selected = (
        df.drop_duplicates(
            subset=["tid"],
            keep="first",
        )
        .copy()
    )

    return selected


# ============================================================
# BALANCED SAMPLING
# ============================================================

def create_balanced_training_set(
    dataframe,
):
    """
    Select a balanced set of independent TIC targets.

    50 positive + 50 negative by default.
    """

    positive = dataframe[
        dataframe["label"] == 1
    ].copy()

    negative = dataframe[
        dataframe["label"] == 0
    ].copy()

    print()
    print(
        "Unique TICs after observation selection:"
    )

    print(
        f"Positive: {len(positive)}"
    )

    print(
        f"Negative: {len(negative)}"
    )

    if len(positive) < TARGETS_PER_CLASS:

        raise ValueError(
            f"Only {len(positive)} positive TICs "
            f"available; need {TARGETS_PER_CLASS}."
        )

    if len(negative) < TARGETS_PER_CLASS:

        raise ValueError(
            f"Only {len(negative)} negative TICs "
            f"available; need {TARGETS_PER_CLASS}."
        )

    # Reproducible random selection.
    positive = positive.sample(
        n=TARGETS_PER_CLASS,
        random_state=RANDOM_STATE,
    )

    negative = negative.sample(
        n=TARGETS_PER_CLASS,
        random_state=RANDOM_STATE,
    )

    selected = pd.concat(
        [
            positive,
            negative,
        ],
        ignore_index=True,
    )

    # Shuffle final dataset.
    selected = selected.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(
        drop=True
    )

    return selected


# ============================================================
# FINAL COLUMN SELECTION
# ============================================================

def prepare_output(
    dataframe,
):
    """
    Keep only the metadata needed for downstream processing.
    """

    preferred_columns = [
        "toi",
        "tid",
        "tfopwg_disp",
        "label",
        "pl_orbper",
        "pl_trandep",
        "pl_trandurh",
        "sequence_number",
        "obs_id",
        "obsid",
        "t_min",
        "t_max",
        "t_exptime",
        "dataRights",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in dataframe.columns
    ]

    return dataframe[
        available_columns
    ].copy()


# ============================================================
# SAVE
# ============================================================

def save_dataset(
    dataframe,
):
    """Save selected observations."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Saved selected training observations:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    dataframe,
):
    """Print final selection summary."""

    print()
    print(
        "=== Selected Training Dataset ==="
    )

    print()

    print(
        f"Total targets: "
        f"{len(dataframe)}"
    )

    print(
        f"Positive: "
        f"{(dataframe['label'] == 1).sum()}"
    )

    print(
        f"Negative: "
        f"{(dataframe['label'] == 0).sum()}"
    )

    print(
        f"Unique TICs: "
        f"{dataframe['tid'].nunique()}"
    )

    print()

    print(
        "Cadence/exposure statistics:"
    )

    if "t_exptime" in dataframe.columns:

        print(
            dataframe[
                "t_exptime"
            ].describe()
        )

    print()

    print(
        "Selected observations:"
    )

    print(
        dataframe[
            [
                "toi",
                "tid",
                "label",
                "tfopwg_disp",
                "pl_orbper",
                "sequence_number",
                "obs_id",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load.
    # --------------------------------------------------------

    manifest = load_manifest()

    print()
    print(
        f"Loaded availability rows: "
        f"{len(manifest)}"
    )

    # --------------------------------------------------------
    # 2. Clean.
    # --------------------------------------------------------

    manifest = clean_manifest(
        manifest
    )

    print(
        f"Rows after cleaning: "
        f"{len(manifest)}"
    )

    # --------------------------------------------------------
    # 3. One observation per TIC.
    # --------------------------------------------------------

    one_per_tic = (
        select_one_observation_per_tic(
            manifest
        )
    )

    print(
        f"One observation per TIC: "
        f"{len(one_per_tic)}"
    )

    # --------------------------------------------------------
    # 4. Balanced selection.
    # --------------------------------------------------------

    selected = (
        create_balanced_training_set(
            one_per_tic
        )
    )

    # --------------------------------------------------------
    # 5. Keep useful metadata.
    # --------------------------------------------------------

    selected = prepare_output(
        selected
    )

    # --------------------------------------------------------
    # 6. Save.
    # --------------------------------------------------------

    save_dataset(
        selected
    )

    # --------------------------------------------------------
    # 7. Summary.
    # --------------------------------------------------------

    print_summary(
        selected
    )


if __name__ == "__main__":
    main()