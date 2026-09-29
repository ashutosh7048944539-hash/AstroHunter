from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
from io import BytesIO

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_FILE = (
    DATA_DIR
    / "ml"
    / "tess_labeled_targets.csv"
)

ARCHIVE_URL = (
    "https://exoplanetarchive.ipac.caltech.edu"
    "/TAP/sync"
)

# Number of targets from each class for the first MVP.
TARGETS_PER_CLASS = 100


# ============================================================
# BUILD NASA EXOPLANET ARCHIVE QUERY
# ============================================================

def build_query():
    """
    Build the TAP query for the NASA Exoplanet Archive
    TESS Project Candidates (TOI) table.

    We retrieve:
        toi
        TIC ID
        TFOPWG disposition
        orbital period
        transit depth
        transit duration

    The last three are metadata for inspection only.
    They will NOT be used as AstroHunter ML features.
    """

    query = """
        SELECT
            toi,
            tid,
            tfopwg_disp,
            pl_orbper,
            pl_trandep,
            pl_trandurh
        FROM toi
        WHERE tfopwg_disp IN (
            'CP',
            'KP',
            'PC',
            'FP',
            'FA'
        )
        AND tid IS NOT NULL
        ORDER BY tid
    """

    # Convert multiline SQL into a single clean query.
    return " ".join(query.split())


# ============================================================
# DOWNLOAD TOI CATALOG
# ============================================================

def download_toi_catalog():
    """
    Download the relevant TOI records from the
    NASA Exoplanet Archive TAP service.
    """

    query = build_query()

    encoded_query = quote(
        query,
        safe="",
    )

    url = (
        f"{ARCHIVE_URL}"
        f"?query={encoded_query}"
        f"&format=csv"
    )

    print()
    print(
        "Downloading TESS candidate catalog..."
    )

    print(
        "Source:"
    )

    print(
        "NASA Exoplanet Archive"
    )

    try:

        with urlopen(
            url,
            timeout=60,
        ) as response:

            raw_data = response.read()

    except Exception as error:

        print()
        print(
            "Failed to download the NASA "
            "Exoplanet Archive catalog."
        )

        print()
        print(
            f"Error: {error}"
        )

        print()
        print(
            "Query used:"
        )

        print(
            query
        )

        raise

    dataframe = pd.read_csv(
        BytesIO(raw_data)
    )

    return dataframe


# ============================================================
# CLEAN CATALOG
# ============================================================

def clean_catalog(dataframe):
    """
    Clean and standardize the downloaded catalog.
    """

    df = dataframe.copy()

    # --------------------------------------------------------
    # Standardize column names.
    # --------------------------------------------------------

    df.columns = [
        column.strip().lower()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Convert TIC ID to numeric.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Clean disposition.
    # --------------------------------------------------------

    df["tfopwg_disp"] = (
        df["tfopwg_disp"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Map dispositions to ML labels.
    #
    # Positive:
    #   CP = Confirmed Planet
    #   KP = Known Planet
    #   PC = Planetary Candidate
    #
    # Negative:
    #   FP = False Positive
    #   FA = False Alarm
    # --------------------------------------------------------

    positive_labels = {
        "CP",
        "KP",
        "PC",
    }

    negative_labels = {
        "FP",
        "FA",
    }

    def assign_label(disposition):

        if disposition in positive_labels:
            return 1

        if disposition in negative_labels:
            return 0

        return None

    df["label"] = (
        df["tfopwg_disp"]
        .map(assign_label)
    )

    # Remove anything that wasn't explicitly mapped.
    df = df.dropna(
        subset=["label"]
    )

    df["label"] = (
        df["label"]
        .astype(int)
    )

    return df


# ============================================================
# ONE TARGET PER TIC
# ============================================================

def select_one_target_per_tic(
    dataframe,
):
    """
    Select one TOI per TIC ID.

    This prevents multiple TOIs belonging to the same
    host star from dominating our small MVP dataset.

    We prefer a positive example if both positive and
    negative dispositions exist for the same TIC.
    """

    selected_rows = []

    for tic_id, group in dataframe.groupby(
        "tid",
        sort=True,
    ):

        positive_rows = group[
            group["label"] == 1
        ]

        if not positive_rows.empty:

            selected_rows.append(
                positive_rows.iloc[0]
            )

        else:

            negative_rows = group[
                group["label"] == 0
            ]

            if not negative_rows.empty:

                selected_rows.append(
                    negative_rows.iloc[0]
                )

    if not selected_rows:

        raise ValueError(
            "No usable TIC targets found."
        )

    result = pd.DataFrame(
        selected_rows
    ).reset_index(
        drop=True
    )

    return result


# ============================================================
# CREATE BALANCED DATASET
# ============================================================

def create_balanced_dataset(
    dataframe,
    targets_per_class,
):
    """
    Create a balanced MVP dataset.

    The first experiment uses the same number of
    positive and negative targets.
    """

    positive = dataframe[
        dataframe["label"] == 1
    ].copy()

    negative = dataframe[
        dataframe["label"] == 0
    ].copy()

    print()
    print(
        "Available unique TIC targets:"
    )

    print(
        f"Positive: {len(positive)}"
    )

    print(
        f"Negative: {len(negative)}"
    )

    if len(positive) < targets_per_class:

        raise ValueError(
            "Not enough positive targets "
            "for requested dataset size."
        )

    if len(negative) < targets_per_class:

        raise ValueError(
            "Not enough negative targets "
            "for requested dataset size."
        )

    # Fixed random seed for reproducibility.
    positive = positive.sample(
        n=targets_per_class,
        random_state=42,
    )

    negative = negative.sample(
        n=targets_per_class,
        random_state=42,
    )

    balanced = pd.concat(
        [
            positive,
            negative,
        ],
        ignore_index=True,
    )

    # Shuffle final dataset.
    balanced = balanced.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(
        drop=True
    )

    return balanced


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(dataframe):
    """
    Save the labeled target manifest.
    """

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
        "Saved dataset to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(dataframe):
    """
    Print a human-readable dataset summary.
    """

    print()
    print(
        "=== Day 4: ML Dataset Summary ==="
    )

    print()

    print(
        f"Total targets: "
        f"{len(dataframe)}"
    )

    print(
        f"Positive targets: "
        f"{(dataframe['label'] == 1).sum()}"
    )

    print(
        f"Negative targets: "
        f"{(dataframe['label'] == 0).sum()}"
    )

    print()

    print(
        "Disposition counts:"
    )

    print(
        dataframe[
            "tfopwg_disp"
        ].value_counts()
    )

    print()

    print(
        "First 10 rows:"
    )

    print(
        dataframe.head(10).to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Download catalog.
    # --------------------------------------------------------

    raw_catalog = (
        download_toi_catalog()
    )

    print()

    print(
        f"Downloaded rows: "
        f"{len(raw_catalog)}"
    )

    # --------------------------------------------------------
    # 2. Clean.
    # --------------------------------------------------------

    clean_data = clean_catalog(
        raw_catalog
    )

    print(
        f"Usable rows after cleaning: "
        f"{len(clean_data)}"
    )

    # --------------------------------------------------------
    # 3. One target per TIC.
    # --------------------------------------------------------

    unique_targets = (
        select_one_target_per_tic(
            clean_data
        )
    )

    print(
        f"Unique TIC targets: "
        f"{len(unique_targets)}"
    )

    # --------------------------------------------------------
    # 4. Balanced MVP dataset.
    # --------------------------------------------------------

    dataset = (
        create_balanced_dataset(
            unique_targets,
            TARGETS_PER_CLASS,
        )
    )

    # --------------------------------------------------------
    # 5. Save.
    # --------------------------------------------------------

    save_dataset(
        dataset
    )

    # --------------------------------------------------------
    # 6. Summary.
    # --------------------------------------------------------

    print_summary(
        dataset
    )


if __name__ == "__main__":
    main()