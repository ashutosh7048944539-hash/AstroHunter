from pathlib import Path

import matplotlib.pyplot as plt
from astropy.io import fits


DATA_FILE = Path(
    "data/raw/mastDownload/TESS/"
    "tess2021175071901-s0040-0000000394050135-0211-s/"
    "tess2021175071901-s0040-0000000394050135-0211-s_lc.fits"
)


def main() -> None:
    with fits.open(DATA_FILE) as hdul:
        data = hdul["LIGHTCURVE"].data

        time = data["TIME"]
        flux = data["PDCSAP_FLUX"]

    plt.figure(figsize=(12, 5))
    plt.plot(time, flux, ".", markersize=1)
    plt.xlabel("Time (BJD - 2457000)")
    plt.ylabel("PDCSAP Flux (e-/s)")
    plt.title("TESS Light Curve — TOI-2025 — Sector 40")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()