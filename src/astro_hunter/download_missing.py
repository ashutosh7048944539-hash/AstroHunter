from pathlib import Path

import pandas as pd
from astroquery.mast import Observations


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

MISSING_TICS = {100267480, 326732851}


def main():
    df = pd.read_csv(INPUT_FILE)

    missing = df[df["tid"].isin(MISSING_TICS)]

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in missing.iterrows():
        tic_id = int(row["tid"])
        obsid = str(int(row["obsid"]))

        print()
        print(f"TIC: {tic_id}")
        print(f"TOI: {row['toi']}")
        print(f"Observation: {row['obs_id']}")
        print(f"MAST obsid: {obsid}")

        products = Observations.get_product_list(obsid)

        lightcurves = [
            product
            for product in products
            if str(product["productFilename"]).lower().endswith("_lc.fits")
        ]

        if not lightcurves:
            print("ERROR: No light-curve FITS found.")
            continue

        product = lightcurves[0]

        print(f"Product: {product['productFilename']}")

        manifest = Observations.download_products(
            product,
            download_dir=str(RAW_DATA_DIR),
            mrp_only=False,
        )

        print("Download complete.")


if __name__ == "__main__":
    main()