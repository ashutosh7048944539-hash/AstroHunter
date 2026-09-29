from pathlib import Path

import pandas as pd
from astroquery.mast import Observations


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "tess_labeled_targets.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "tess_available_targets.csv"
)

# Our current single-sector pipeline works best when
# multiple transit events can fit inside one sector.
MAX_PERIOD_DAYS = 14.0

# Query MAST in batches instead of sending hundreds of
# TIC IDs in one request.
BATCH_SIZE = 25


# ============================================================
# LOAD TARGET CATALOG
# ============================================================

def load_target_catalog():
    """Load the labeled target manifest."""

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = {
        "toi",
        "tid",
        "tfopwg_disp",
        "label",
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
# FILTER BY PERIOD
# ============================================================

def filter_by_period(dataframe):
    """
    Keep targets whose catalog period is short enough
    for the current single-sector MVP pipeline.
    """

    df = dataframe.copy()

    # Some rows may have missing periods.
    df["pl_orbper"] = pd.to_numeric(
        df["pl_orbper"],
        errors="coerce",
    )

    before = len(df)

    df = df[
        df["pl_orbper"].notna()
        & (
            df["pl_orbper"]
            <= MAX_PERIOD_DAYS
        )
    ].copy()

    after = len(df)

    print()
    print(
        "=== Period filter ==="
    )

    print(
        f"Before: {before}"
    )

    print(
        f"After:  {after}"
    )

    print(
        f"Removed: {before - after}"
    )

    return df


# ============================================================
# MAST QUERY
# ============================================================

def query_mast_batch(tic_ids):
    """
    Query MAST for TESS/SPOC time-series observations
    for a batch of TIC IDs.
    """

    # MAST expects target names as strings.
    tic_names = [
        str(int(tic_id))
        for tic_id in tic_ids
    ]

    observations = (
        Observations.query_criteria(
            obs_collection="TESS",
            provenance_name="SPOC",
            target_name=tic_names,
            dataproduct_type="timeseries",
        )
    )

    return observations


# ============================================================
# EXTRACT OBSERVATION INFORMATION
# ============================================================

def observations_to_dataframe(
    observations,
):
    """
    Convert the Astroquery Table returned by MAST
    into a pandas DataFrame containing the fields
    needed for our availability manifest.
    """

    if len(observations) == 0:

        return pd.DataFrame()

    rows = []

    for row in observations:

        rows.append(
            {
                "tid": int(
                    row["target_name"]
                ),
                "obs_id": str(
                    row["obs_id"]
                ),
                "obsid": str(
                    row["obsid"]
                ),
                "sequence_number": (
                    row["sequence_number"]
                ),
                "t_min": row["t_min"],
                "t_max": row["t_max"],
                "t_exptime": (
                    row["t_exptime"]
                ),
                "dataRights": str(
                    row["dataRights"]
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# QUERY ALL TARGETS
# ============================================================

def find_available_targets(
    dataframe,
):
    """
    Query MAST in batches and build a manifest of
    available TESS/SPOC observations.
    """

    all_results = []

    tic_ids = (
        dataframe["tid"]
        .astype("int64")
        .tolist()
    )

    total = len(tic_ids)

    print()
    print(
        "=== MAST availability search ==="
    )

    print(
        f"Targets to check: {total}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):

        end = min(
            start + BATCH_SIZE,
            total,
        )

        batch = tic_ids[
            start:end
        ]

        batch_number = (
            start // BATCH_SIZE
        ) + 1

        total_batches = (
            (total + BATCH_SIZE - 1)
            // BATCH_SIZE
        )

        print()
        print(
            f"Batch "
            f"{batch_number}/{total_batches} "
            f"({start + 1}-{end})"
        )

        try:

            observations = (
                query_mast_batch(
                    batch
                )
            )

        except Exception as error:

            print(
                "MAST query failed:"
            )

            print(error)

            print(
                "Skipping this batch."
            )

            continue

        batch_results = (
            observations_to_dataframe(
                observations
            )
        )

        if not batch_results.empty:

            all_results.append(
                batch_results
            )

            print(
                f"Observations found: "
                f"{len(batch_results)}"
            )

        else:

            print(
                "No observations found."
            )

    if not all_results:

        return pd.DataFrame()

    return pd.concat(
        all_results,
        ignore_index=True,
    )


# ============================================================
# MERGE LABELS + OBSERVATIONS
# ============================================================

def build_availability_manifest(
    targets,
    observations,
):
    """
    Merge the labeled target catalog with MAST
    observation metadata.
    """

    if observations.empty:

        raise ValueError(
            "No usable TESS/SPOC observations "
            "were found."
        )

    # Keep only targets for which MAST returned
    # an actual observation.
    available_tics = (
        observations["tid"]
        .astype("int64")
        .unique()
    )

    available_targets = targets[
        targets["tid"].isin(
            available_tics
        )
    ].copy()

    manifest = available_targets.merge(
        observations,
        on="tid",
        how="inner",
    )

    return manifest


# ============================================================
# SAVE
# ============================================================

def save_manifest(
    dataframe,
):
    """Save the MAST availability manifest."""

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
        "Saved availability manifest:"
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
    """Print availability statistics."""

    print()
    print(
        "=== MAST Availability Summary ==="
    )

    print()

    unique_targets = (
        dataframe["tid"]
        .nunique()
    )

    unique_sectors = (
        dataframe[
            "sequence_number"
        ].nunique()
    )

    print(
        f"Unique usable TICs: "
        f"{unique_targets}"
    )

    print(
        f"Unique sectors: "
        f"{unique_sectors}"
    )

    print()

    print(
        "Class distribution:"
    )

    print(
        dataframe[
            ["tid", "label"]
        ]
        .drop_duplicates()
        ["label"]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Observations by sector:"
    )

    print(
        dataframe[
            "sequence_number"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "Example observations:"
    )

    print(
        dataframe[
            [
                "toi",
                "tid",
                "label",
                "pl_orbper",
                "sequence_number",
                "obs_id",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load labeled target manifest.
    # --------------------------------------------------------

    targets = (
        load_target_catalog()
    )

    print()
    print(
        f"Loaded targets: "
        f"{len(targets)}"
    )

    # --------------------------------------------------------
    # 2. Filter periods.
    # --------------------------------------------------------

    targets = (
        filter_by_period(
            targets
        )
    )

    # --------------------------------------------------------
    # 3. Query MAST.
    # --------------------------------------------------------

    observations = (
        find_available_targets(
            targets
        )
    )

    # --------------------------------------------------------
    # 4. Merge labels and observations.
    # --------------------------------------------------------

    manifest = (
        build_availability_manifest(
            targets,
            observations,
        )
    )

    # --------------------------------------------------------
    # 5. Save.
    # --------------------------------------------------------

    save_manifest(
        manifest
    )

    # --------------------------------------------------------
    # 6. Print summary.
    # --------------------------------------------------------

    print_summary(
        manifest
    )


if __name__ == "__main__":
    main()