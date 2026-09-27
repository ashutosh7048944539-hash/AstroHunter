from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation


DATA_FILE = Path(
    "data/raw/mastDownload/TESS/"
    "tess2021175071901-s0040-0000000394050135-0211-s/"
    "tess2021175071901-s0040-0000000394050135-0211-s_lc.fits"
)


def load_lightcurve():
    """Load the TESS light-curve measurements."""
    with fits.open(DATA_FILE) as hdul:
        data = hdul["LIGHTCURVE"].data

        time = np.asarray(data["TIME"], dtype=float)
        flux = np.asarray(data["PDCSAP_FLUX"], dtype=float)
        flux_err = np.asarray(data["PDCSAP_FLUX_ERR"], dtype=float)
        quality = np.asarray(data["QUALITY"])

    return time, flux, flux_err, quality


def filter_lightcurve(time, flux, flux_err, quality):
    """Remove invalid measurements and quality-flagged cadences."""
    finite_mask = np.isfinite(time) & np.isfinite(flux)
    quality_mask = quality == 0
    combined_mask = finite_mask & quality_mask

    clean_time = time[combined_mask]
    clean_flux = flux[combined_mask]
    clean_flux_err = flux_err[combined_mask]

    return clean_time, clean_flux, clean_flux_err


def normalize_flux(flux):
    """Normalize flux using its median baseline."""
    baseline = np.median(flux)
    normalized_flux = flux / baseline

    return normalized_flux, baseline


def detrend_flux(flux, time_differences):
    """Remove long-term baseline variations from normalized flux."""
    cadence_days = np.median(time_differences)
    cadence_minutes = cadence_days * 24 * 60

    window_minutes = 24 * 60
    window_points = int(window_minutes / cadence_minutes)

    # Median filters work best with an odd-sized window.
    if window_points % 2 == 0:
        window_points += 1

    trend = median_filter(
        flux,
        size=window_points,
        mode="nearest",
    )

    detrended_flux = flux / trend

    return detrended_flux, cadence_minutes, window_minutes, window_points


def diagnose_outliers(flux):
    """Diagnose potential outliers using a robust local baseline."""
    local_baseline = median_filter(
        flux,
        size=31,
        mode="nearest",
    )

    residuals = flux - local_baseline

    robust_sigma = 1.4826 * median_abs_deviation(
        residuals,
        scale=1.0,
    )

    outlier_threshold = 5 * robust_sigma
    outlier_mask = np.abs(residuals) > outlier_threshold

    positive_outliers = outlier_mask & (residuals > 0)
    negative_outliers = outlier_mask & (residuals < 0)

    return (
        robust_sigma,
        outlier_threshold,
        outlier_mask,
        positive_outliers,
        negative_outliers,
    )


def main() -> None:
    time, flux, flux_err, quality = load_lightcurve()

    # Step 1: remove invalid measurements and flagged cadences.
    clean_time, clean_flux, clean_flux_err = filter_lightcurve(
        time,
        flux,
        flux_err,
        quality,
    )

    time_differences = np.diff(clean_time)

    gap_threshold = 3 * np.median(time_differences)
    large_gaps = time_differences > gap_threshold

    print()
    print("=== Cadence gaps ===")
    print(
        f"Median gap:              "
        f"{np.median(time_differences) * 24 * 60:.2f} minutes"
    )
    print(
        f"Largest gap:             "
        f"{np.max(time_differences) * 24 * 60:.2f} minutes"
    )
    print(
        f"Large gaps (>3x median): "
        f"{large_gaps.sum():,}"
    )

    print()
    print("=== Cadence check ===")
    print(
        f"Median cadence: "
        f"{np.median(time_differences) * 24 * 60:.2f} minutes"
    )
    print(
        f"Median cadence: "
        f"{np.median(time_differences) * 24 * 3600:.2f} seconds"
    )

    # Step 2: normalize flux by the median.
    normalized_flux, baseline = normalize_flux(clean_flux)

    # Step 3: remove long-term baseline variations.
    detrended_flux, cadence_minutes, window_minutes, window_points = (
        detrend_flux(normalized_flux, time_differences)
    )

    print()
    print("=== Detrending ===")
    print(f"Cadence:                 {cadence_minutes:.1f} minutes")
    print(f"Window:                  {window_minutes:.0f} minutes")
    print(f"Window size:             {window_points} cadences")
    print(f"Detrended median:        {np.median(detrended_flux):.6f}")
    print(f"Detrended minimum:       {detrended_flux.min():.6f}")
    print(f"Detrended maximum:       {detrended_flux.max():.6f}")

    # Step 4: diagnose potential outliers.
    # We do not remove them yet.
    (
        robust_sigma,
        outlier_threshold,
        outlier_mask,
        positive_outliers,
        negative_outliers,
    ) = diagnose_outliers(detrended_flux)

    print()
    print("=== Outlier diagnostics ===")
    print(f"Robust scatter:          {robust_sigma:.6f}")
    print(f"5-sigma threshold:       {outlier_threshold:.6f}")
    print(f"Potential outliers:      {outlier_mask.sum():,}")
    print(f"Positive outliers:       {positive_outliers.sum():,}")
    print(f"Negative outliers:       {negative_outliers.sum():,}")

    print()
    print("=== Day 2: Preprocessing summary ===")
    print(f"Usable measurements:     {len(clean_time):,}")
    print(f"Median flux:             {baseline:.3f} e-/s")
    print(f"Normalized minimum:      {normalized_flux.min():.6f}")
    print(f"Normalized maximum:      {normalized_flux.max():.6f}")
    print(f"Normalized median:       {np.median(normalized_flux):.6f}")

    # Plot detrended light curve.
    plt.figure(figsize=(12, 5))
    plt.plot(clean_time, detrended_flux, ".", markersize=1)

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
        label="Detrended baseline",
    )

    plt.xlabel("Time (BJD - 2457000)")
    plt.ylabel("Detrended flux")
    plt.title("Detrended TESS Light Curve — TOI-2025 — Sector 40")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()