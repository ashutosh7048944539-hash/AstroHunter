from pathlib import Path
import sys

import numpy as np
import pandas as pd

from astro_hunter.feature_extraction import extract_features
from astro_hunter.preprocess import preprocess_lightcurve


RAW_DIR = Path("data/raw/ml")
METADATA_FILE = Path("data/ml/selected_training_observations.csv")
OUTPUT_FILE = Path("data/ml/extracted_features.csv")


def find_fits_for_tic(tic: str) -> Path | None:
    """
    Find the FITS file corresponding to a TIC ID.
    """

    matches = list(
        RAW_DIR.rglob(f"*{tic}*_lc.fits")
    )

    if not matches:
        return None

    if len(matches) > 1:
        print(
            f"       WARNING: Multiple FITS files found "
            f"for TIC {tic}"
        )
        print(
            f"       Using: {matches[0].name}"
        )

    return matches[0]


def empty_features() -> dict:
    """
    Return an empty feature dictionary.

    NaN means that the feature could not be
    reliably measured.
    """

    return {
        "period_days": np.nan,
        "transit_depth": np.nan,
        "duration_hours": np.nan,
        "num_transit_points": np.nan,
        "baseline_scatter": np.nan,
        "transit_snr": np.nan,
        "num_events": np.nan,
        "period_scatter_days": np.nan,
    }


def extract_from_fits(
    fits_path: Path,
) -> dict:
    """
    Run the same preprocessing + feature extraction
    pipeline used for individual targets.
    """

    (
        time,
        detrended_flux,
        flux_err,
        baseline,
        cadence_minutes,
        window_points,
    ) = preprocess_lightcurve(fits_path)

    print(
        f"       Cadence: {cadence_minutes:.2f} minutes"
    )

    print(
        f"       Detrend window: {window_points} points"
    )

    features = extract_features(
        time,
        detrended_flux,
    )

    return features


def safe_extract_from_fits(
    fits_path: Path,
) -> tuple[dict, str]:

    try:

        features = extract_from_fits(
            fits_path
        )

        return features, "success"

    except ValueError as exc:

        print(
            f"       No usable candidate signal: "
            f"{exc}"
        )

        return empty_features(), "no_candidate_events"

    except Exception as exc:

        print(
            f"       ERROR extracting:"
        )

        print(
            f"       {type(exc).__name__}: {exc}"
        )

        return empty_features(), "failed"


def main() -> None:

    print(
        "=== Day 4: Batch Feature Extraction ==="
    )

    print()

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not METADATA_FILE.exists():

        print(
            f"ERROR: Metadata file not found:"
        )

        print(
            f"       {METADATA_FILE}"
        )

        sys.exit(1)

    if not RAW_DIR.exists():

        print(
            f"ERROR: Raw data directory not found:"
        )

        print(
            f"       {RAW_DIR}"
        )

        sys.exit(1)

    metadata = pd.read_csv(
        METADATA_FILE
    )

    print(
        f"Metadata rows: {len(metadata)}"
    )

    print()

    # --------------------------------------------------------
    # Verify expected schema
    # --------------------------------------------------------

    required_columns = [
        "toi",
        "tid",
        "tfopwg_disp",
        "label",
        "pl_orbper",
        "obs_id",
        "obsid",
        "sequence_number",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metadata.columns
    ]

    if missing_columns:

        print(
            "ERROR: Missing metadata columns:"
        )

        for column in missing_columns:

            print(
                f"       {column}"
            )

        sys.exit(1)

    # --------------------------------------------------------
    # Process every target
    # --------------------------------------------------------

    rows = []

    for index, row in metadata.iterrows():

        tic = str(
            int(row["tid"])
        )

        print(
            f"[{index + 1:3d}/{len(metadata)}] "
            f"TIC {tic} | "
            f"TOI {row['toi']}"
        )

        # ----------------------------------------------------
        # Locate FITS
        # ----------------------------------------------------

        fits_path = find_fits_for_tic(
            tic
        )

        if fits_path is None:

            print(
                "       ERROR: FITS file not found"
            )

            features = empty_features()

            extraction_status = (
                "missing_fits"
            )

        else:

            print(
                f"       FITS: "
                f"{fits_path.name}"
            )

            features, extraction_status = (
                safe_extract_from_fits(
                    fits_path
                )
            )

        # ----------------------------------------------------
        # Build output row
        # ----------------------------------------------------

        output_row = {

            # Metadata
            "tic_id": int(tic),

            "toi": row["toi"],

            "label": int(
                row["label"]
            ),

            "tfopwg_disp": row[
                "tfopwg_disp"
            ],

            "archive_period_days": row[
                "pl_orbper"
            ],

            "sequence_number": row[
                "sequence_number"
            ],

            "obs_id": row[
                "obs_id"
            ],

            "obsid": row[
                "obsid"
            ],

            "fits_path": (
                str(fits_path)
                if fits_path
                else ""
            ),

            "extraction_status":
                extraction_status,

            # AstroHunter features
            **features,
        }

        rows.append(
            output_row
        )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    result = pd.DataFrame(
        rows
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()

    print(
        "=== Batch Extraction Summary ==="
    )

    print(
        f"Total targets:          "
        f"{len(result)}"
    )

    print(
        f"Successful extraction: "
        f"{(result['extraction_status'] == 'success').sum()}"
    )

    print(
        f"No candidate events:    "
        f"{(result['extraction_status'] == 'no_candidate_events').sum()}"
    )

    print(
        f"Missing FITS:           "
        f"{(result['extraction_status'] == 'missing_fits').sum()}"
    )

    print(
        f"Failed extraction:      "
        f"{(result['extraction_status'] == 'failed').sum()}"
    )

    # --------------------------------------------------------
    # Missingness
    # --------------------------------------------------------

    print()

    print(
        "=== Feature Missingness ==="
    )

    feature_columns = [

        "period_days",

        "transit_depth",

        "duration_hours",

        "num_transit_points",

        "baseline_scatter",

        "transit_snr",

        "num_events",

        "period_scatter_days",
    ]

    for column in feature_columns:

        missing = (
            result[column]
            .isna()
            .sum()
        )

        print(
            f"{column:25s}: "
            f"{missing:3d} missing / "
            f"{len(result)}"
        )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print()

    print(
        "=== Class Distribution ==="
    )

    print(
        result["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    print()

    print(
        "=== Feature Table Preview ==="
    )

    preview_columns = [

        "tic_id",

        "label",

        "transit_depth",

        "duration_hours",

        "num_transit_points",

        "baseline_scatter",

        "transit_snr",

        "num_events",

        "period_days",

        "period_scatter_days",
    ]

    print(
        result[
            preview_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":

    main()