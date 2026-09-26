# AstroHunter Data

## Primary dataset

AstroHunter currently uses a TESS Sector 40 light curve for
TOI-2025 (TIC 394050135).

### Source

- Mission: NASA TESS
- Archive: Mikulski Archive for Space Telescopes (MAST)
- Data product: TESS Science Processing Operations Center (SPOC) light curve
- Sector: 40
- Cadence: 120 seconds
- Target: TIC 394050135
- MAST observation ID: 62364120
- Product:
  `tess2021175071901-s0040-0000000394050135-0211-s_lc.fits`

### File format

The light curve is provided as a FITS file.

The primary light-curve table contains measurements including:

- `TIME` — observation time in BJD - 2457000 days
- `SAP_FLUX` — simple aperture photometry flux
- `SAP_FLUX_ERR` — uncertainty in SAP flux
- `PDCSAP_FLUX` — systematics-corrected flux
- `PDCSAP_FLUX_ERR` — uncertainty in PDCSAP flux
- `QUALITY` — TESS data-quality bitmask

### Current raw-data baseline

- Total rows: 20,309
- Valid time values: 19,643
- Valid flux values: 19,611
- Observation duration: 28.206 days
- Median PDCSAP flux: 6872.175 e-/s
- QUALITY == 0: 19,610

## Reproducibility

The raw FITS file is obtained programmatically from MAST using
Astroquery.

Raw downloaded data are kept locally under:

`data/raw/`

and are not intended to be committed to the Git repository.