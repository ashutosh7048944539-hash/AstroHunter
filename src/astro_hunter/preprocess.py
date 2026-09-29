from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation


def load_lightcurve(data_file: Path):
    """Load a TESS light curve from a FITS file."""

    with fits.open(data_file) as hdul:
        data = hdul["LIGHTCURVE"].data

        time = np.asarray(
            data["TIME"],
            dtype=float,
        )

        flux = np.asarray(
            data["PDCSAP_FLUX"],
            dtype=float,
        )

        flux_err = np.asarray(
            data["PDCSAP_FLUX_ERR"],
            dtype=float,
        )

        quality = np.asarray(
            data["QUALITY"],
        )

    return time, flux, flux_err, quality


def filter_lightcurve(
    time,
    flux,
    flux_err,
    quality,
):
    """Remove invalid measurements and quality-flagged cadences."""

    finite_mask = (
        np.isfinite(time)
        & np.isfinite(flux)
    )

    quality_mask = (
        quality == 0
    )

    combined_mask = (
        finite_mask
        & quality_mask
    )

    clean_time = time[combined_mask]
    clean_flux = flux[combined_mask]
    clean_flux_err = flux_err[combined_mask]

    return (
        clean_time,
        clean_flux,
        clean_flux_err,
    )


def normalize_flux(flux):
    """Normalize flux using its median baseline."""

    baseline = np.median(flux)

    normalized_flux = (
        flux / baseline
    )

    return normalized_flux, baseline


def detrend_flux(
    flux,
    time_differences,
):
    """Remove long-term baseline variations."""

    cadence_days = np.median(
        time_differences
    )

    cadence_minutes = (
        cadence_days * 24 * 60
    )

    window_minutes = 24 * 60

    window_points = int(
        window_minutes / cadence_minutes
    )

    if window_points < 3:
        window_points = 3

    if window_points % 2 == 0:
        window_points += 1

    trend = median_filter(
        flux,
        size=window_points,
        mode="nearest",
    )

    detrended_flux = (
        flux / trend
    )

    return (
        detrended_flux,
        cadence_minutes,
        window_minutes,
        window_points,
    )


def diagnose_outliers(flux):
    """Diagnose potential outliers without removing them."""

    local_baseline = median_filter(
        flux,
        size=31,
        mode="nearest",
    )

    residuals = (
        flux - local_baseline
    )

    robust_sigma = (
        1.4826
        * median_abs_deviation(
            residuals,
            scale=1.0,
        )
    )

    outlier_threshold = (
        5 * robust_sigma
    )

    outlier_mask = (
        np.abs(residuals)
        > outlier_threshold
    )

    positive_outliers = (
        outlier_mask
        & (residuals > 0)
    )

    negative_outliers = (
        outlier_mask
        & (residuals < 0)
    )

    return (
        robust_sigma,
        outlier_threshold,
        outlier_mask,
        positive_outliers,
        negative_outliers,
    )


def preprocess_lightcurve(data_file: Path):
    """
    Load and preprocess one TESS light curve.

    Returns:
        clean_time
        detrended_flux
        clean_flux_err
        baseline
        cadence_minutes
        window_points
    """

    (
        time,
        flux,
        flux_err,
        quality,
    ) = load_lightcurve(data_file)

    (
        clean_time,
        clean_flux,
        clean_flux_err,
    ) = filter_lightcurve(
        time,
        flux,
        flux_err,
        quality,
    )

    if len(clean_time) < 10:
        raise ValueError(
            "Not enough valid measurements."
        )

    time_differences = np.diff(
        clean_time
    )

    if len(time_differences) == 0:
        raise ValueError(
            "Unable to calculate cadence."
        )

    (
        normalized_flux,
        baseline,
    ) = normalize_flux(
        clean_flux
    )

    (
        detrended_flux,
        cadence_minutes,
        _,
        window_points,
    ) = detrend_flux(
        normalized_flux,
        time_differences,
    )

    return (
        clean_time,
        detrended_flux,
        clean_flux_err,
        baseline,
        cadence_minutes,
        window_points,
    )