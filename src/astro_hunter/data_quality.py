from pathlib import Path

import numpy as np
from astropy.io import fits


DATA_FILE = Path(
    "data/raw/mastDownload/TESS/"
    "tess2021175071901-s0040-0000000394050135-0211-s/"
    "tess2021175071901-s0040-0000000394050135-0211-s_lc.fits"
)


def main() -> None:
    with fits.open(DATA_FILE) as hdul:
        data = hdul["LIGHTCURVE"].data

        time = np.asarray(data["TIME"])
        flux = np.asarray(data["PDCSAP_FLUX"])
        quality = np.asarray(data["QUALITY"])

    valid_time = np.isfinite(time)
    valid_flux = np.isfinite(flux)
    valid_both = valid_time & valid_flux

    print("=== AstroHunter Data Quality Report ===")
    print(f"Total rows:              {len(time):,}")
    print(f"Valid TIME values:       {valid_time.sum():,}")
    print(f"Valid flux values:       {valid_flux.sum():,}")
    print(f"Valid TIME + flux:       {valid_both.sum():,}")

    print()
    print(f"NaN TIME values:         {(~valid_time).sum():,}")
    print(f"NaN flux values:         {(~valid_flux).sum():,}")

    print()
    print(f"Flux minimum:            {np.nanmin(flux):.3f} e-/s")
    print(f"Flux maximum:            {np.nanmax(flux):.3f} e-/s")
    print(f"Flux median:             {np.nanmedian(flux):.3f} e-/s")

    print()
    print(f"Observation start:       {np.nanmin(time):.6f}")
    print(f"Observation end:         {np.nanmax(time):.6f}")
    print(
        f"Observation duration:    "
        f"{np.nanmax(time) - np.nanmin(time):.3f} days"
    )

    print()
    print(f"QUALITY == 0:            {(quality == 0).sum():,}")
    print(f"QUALITY != 0:            {(quality != 0).sum():,}")


if __name__ == "__main__":
    main()