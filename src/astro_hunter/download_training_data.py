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
    / "selected_training_observations.csv"
)

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ml"
)


# ============================================================
# LOAD SELECTED OBSERVATIONS
# ============================================================

def load_selected_observations():
    """Load the selected training observation manifest."""

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
        "obs_id",
        "obsid",
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
# FIND LIGHT CURVE PRODUCT
# ============================================================

def find_lightcurve_product(obsid):
    """
    Find the TESS/SPOC light-curve FITS product
    associated with a MAST observation group.

    MAST obsid may be loaded from pandas as numpy.int64,
    so explicitly convert it to a normal Python string.
    """

    # --------------------------------------------------------
    # IMPORTANT:
    # astroquery MAST expects obsid values as strings.
    # --------------------------------------------------------

    obsid_string = str(
        int(obsid)
    )

    products = (
        Observations.get_product_list(
            obsid_string
        )
    )

    if len(products) == 0:

        return None

    # --------------------------------------------------------
    # Search for *_lc.fits.
    # --------------------------------------------------------

    for row in products:

        filename = str(
            row["productFilename"]
        )

        if filename.lower().endswith(
            "_lc.fits"
        ):

            return row

    return None


# ============================================================
# DOWNLOAD ONE TARGET
# ============================================================

def download_one_target(
    row,
    index,
    total,
):
    """
    Find and download the SPOC light curve
    for one selected target.
    """

    tic_id = int(
        row["tid"]
    )

    toi = row["toi"]

    obsid = row["obsid"]

    obs_id = row["obs_id"]

    label = int(
        row["label"]
    )

    print()
    print(
        f"[{index}/{total}] "
        f"TIC {tic_id} | "
        f"TOI {toi}"
    )

    print(
        f"Label: {label}"
    )

    print(
        f"Observation: {obs_id}"
    )

    print(
        f"MAST obsid: {obsid}"
    )

    # --------------------------------------------------------
    # Find product.
    # --------------------------------------------------------

    try:

        product = (
            find_lightcurve_product(
                obsid
            )
        )

    except Exception as error:

        print(
            "Failed while querying MAST products:"
        )

        print(error)

        return False

    if product is None:

        print(
            "No light-curve FITS product found."
        )

        return False

    filename = str(
        product["productFilename"]
    )

    print(
        f"Product: {filename}"
    )

    # --------------------------------------------------------
    # Download.
    # --------------------------------------------------------

    try:

        manifest = (
            Observations.download_products(
                product,
                download_dir=str(
                    RAW_DATA_DIR
                ),
                mrp_only=False,
            )
        )

    except Exception as error:

        print(
            "Download failed:"
        )

        print(error)

        return False

    if manifest is None:

        print(
            "MAST returned no download manifest."
        )

        return False

    print(
        "Download complete."
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    dataframe = (
        load_selected_observations()
    )

    total = len(dataframe)

    print()
    print(
        "=== AstroHunter ML Data Download ==="
    )

    print()

    print(
        f"Targets: {total}"
    )

    print(
        "Download directory:"
    )

    print(
        RAW_DATA_DIR
    )

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process every selected target.
    # --------------------------------------------------------

    for index, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):

        success = download_one_target(
            row,
            index,
            total,
        )

        if success:

            successful += 1

        else:

            failed += 1

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    print()
    print(
        "=== Download Summary ==="
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        f"Total:      {total}"
    )


if __name__ == "__main__":
    main()