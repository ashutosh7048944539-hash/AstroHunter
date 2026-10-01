from pathlib import Path

import numpy as np


# ============================================================
# Configuration
# ============================================================

MIN_PERIOD_DAYS = 0.5
MAX_PERIOD_DAYS = 14.0

# Coarse search:
# Used to find the approximate period over the whole search range.
COARSE_PERIOD_STEP = 0.02
COARSE_PHASE_BINS = 300

# Fine search:
# Used around the best coarse period.
FINE_PERIOD_HALF_WIDTH = 0.05
FINE_PERIOD_STEP = 0.0002
FINE_PHASE_BINS = 400

# Candidate event detection
CANDIDATE_THRESHOLD = 0.997
EVENT_GAP_MINUTES = 10.0
STRONG_EVENT_MIN_POINTS = 20

# Event support around BLS ephemeris.
# An event is considered consistent if its midpoint is within
# this fraction of the candidate period from the nearest epoch.
EVENT_PHASE_TOLERANCE = 0.15


# ============================================================
# Candidate event detection
# ============================================================

def group_candidate_points(
    time: np.ndarray,
    candidate_mask: np.ndarray,
    max_gap: float,
) -> list[tuple[int, int]]:
    """
    Group consecutive candidate points into events.

    Parameters
    ----------
    time:
        Time array in days.

    candidate_mask:
        Boolean array indicating candidate transit-like points.

    max_gap:
        Maximum allowed time gap between consecutive candidate
        points, in days.

    Returns
    -------
    list of (start_index, end_index)
    """

    candidate_indices = np.flatnonzero(candidate_mask)

    if len(candidate_indices) == 0:
        return []

    events = []

    start = candidate_indices[0]
    previous = candidate_indices[0]

    for index in candidate_indices[1:]:
        time_gap = time[index] - time[previous]

        if time_gap <= max_gap:
            previous = index
        else:
            events.append((start, previous))
            start = index
            previous = index

    events.append((start, previous))

    return events


def measure_candidate_events(
    time: np.ndarray,
    flux: np.ndarray,
    events: list[tuple[int, int]],
) -> list[dict]:
    """
    Measure basic properties of candidate events.
    """

    measurements = []

    for start, end in events:

        event_time = time[start:end + 1]
        event_flux = flux[start:end + 1]

        if len(event_time) == 0:
            continue

        midpoint = 0.5 * (event_time[0] + event_time[-1])

        duration_days = event_time[-1] - event_time[0]
        duration_hours = duration_days * 24.0

        minimum_flux = np.min(event_flux)
        depth = 1.0 - minimum_flux

        measurements.append(
            {
                "start_index": int(start),
                "end_index": int(end),
                "midpoint": float(midpoint),
                "duration": float(duration_days),
                "duration_hours": float(duration_hours),
                "minimum_flux": float(minimum_flux),
                "depth": float(depth),
                "num_points": int(len(event_time)),
            }
        )

    return measurements


# ============================================================
# BLS-style search
# ============================================================

def bin_phase_folded_lightcurve(
    phase: np.ndarray,
    flux: np.ndarray,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin a phase-folded light curve.

    Returns
    -------
    bin_centers
    bin_means
    bin_counts
    """

    edges = np.linspace(-0.5, 0.5, num_bins + 1)

    bin_indices = np.digitize(phase, edges) - 1

    valid = (
        (bin_indices >= 0)
        & (bin_indices < num_bins)
        & np.isfinite(flux)
    )

    sums = np.bincount(
        bin_indices[valid],
        weights=flux[valid],
        minlength=num_bins,
    )

    counts = np.bincount(
        bin_indices[valid],
        minlength=num_bins,
    )

    means = np.full(num_bins, np.nan)

    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]

    centers = 0.5 * (edges[:-1] + edges[1:])

    return centers, means, counts

def phase_fold_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    reference_time: float | None = None,
    reference_phase: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Phase fold a light curve around a trial period.

    Parameters
    ----------
    time:
        Observation times.

    flux:
        Corresponding flux values.

    period:
        Trial/detected period in days.

    reference_time:
        Time corresponding to phase zero.
        If None, time[0] is used.

    reference_phase:
        Optional phase to shift to zero before folding.
        This is useful when BLS provides the location of
        the detected transit.

    Returns
    -------
    phase:
        Phase values in [-0.5, 0.5).

    flux:
        Unchanged flux values.
    """

    if reference_time is None:
        reference_time = time[0]

    phase = (
        (time - reference_time) / period
        + 0.5
    ) % 1.0 - 0.5

    # Shift the supplied reference phase to phase = 0.
    if reference_phase is not None:
        phase = (
            phase - reference_phase + 0.5
        ) % 1.0 - 0.5

    return phase, flux

def _robust_scatter(flux: np.ndarray) -> float:
    """
    Robust estimate of scatter using MAD.
    """

    finite_flux = flux[np.isfinite(flux)]

    if len(finite_flux) == 0:
        return np.nan

    median = np.median(finite_flux)

    mad = np.median(np.abs(finite_flux - median))

    scatter = 1.4826 * mad

    # Prevent division by zero for exceptionally flat arrays.
    if scatter <= 1e-10:
        scatter = np.std(finite_flux)

    return float(scatter)


def search_candidate_period(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float,
    max_period: float,
    period_step: float,
    num_phase_bins: int,
    duration_hours_grid: np.ndarray | None = None,
) -> dict:
    """
    Fast simplified BLS-style period search.

    The score is:
        depth / scatter * sqrt(N_in * N_out / N_total)

    This penalizes very narrow boxes that contain only a few
    unusually low points.
    """

    if duration_hours_grid is None:
        duration_hours_grid = np.arange(0.5, 8.01, 0.5)

    finite = np.isfinite(time) & np.isfinite(flux)
    time = time[finite]
    flux = flux[finite]

    if len(time) < 100:
        return {
            "best_period": np.nan,
            "best_score": np.nan,
            "best_duration_hours": np.nan,
            "best_phase": np.nan,
            "periods": np.array([]),
            "scores": np.array([]),
        }

    periods = np.arange(
        min_period,
        max_period + period_step / 2.0,
        period_step,
    )

    scores = np.full(len(periods), -np.inf, dtype=float)

    best_period = np.nan
    best_score = -np.inf
    best_duration_hours = np.nan
    best_phase = np.nan

    robust_scatter = _robust_scatter(flux)

    if not np.isfinite(robust_scatter) or robust_scatter <= 0:
        return {
            "best_period": np.nan,
            "best_score": np.nan,
            "best_duration_hours": np.nan,
            "best_phase": np.nan,
            "periods": periods,
            "scores": scores,
        }

    total_points = len(flux)
    total_flux = np.sum(flux)

    for period_index, period in enumerate(periods):

        phase = ((time - time[0]) / period) % 1.0

        bin_index = (
            phase * num_phase_bins
        ).astype(np.int64)

        bin_index = np.clip(
            bin_index,
            0,
            num_phase_bins - 1,
        )

        bin_flux = np.bincount(
            bin_index,
            weights=flux,
            minlength=num_phase_bins,
        )

        bin_counts = np.bincount(
            bin_index,
            minlength=num_phase_bins,
        ).astype(float)

        doubled_flux = np.concatenate(
            [bin_flux, bin_flux]
        )
        doubled_counts = np.concatenate(
            [bin_counts, bin_counts]
        )

        cumulative_flux = np.concatenate(
            [[0.0], np.cumsum(doubled_flux)]
        )
        cumulative_counts = np.concatenate(
            [[0.0], np.cumsum(doubled_counts)]
        )

        starts = np.arange(
            num_phase_bins,
            dtype=np.int64,
        )

        local_best_score = -np.inf
        local_best_duration = np.nan
        local_best_phase = np.nan

        for duration_hours in duration_hours_grid:

            duration_phase = (
                duration_hours / 24.0
            ) / period

            if duration_phase <= 0 or duration_phase >= 0.5:
                continue

            window_bins = max(
                1,
                int(
                    round(
                        duration_phase
                        * num_phase_bins
                    )
                ),
            )

            window_bins = min(
                window_bins,
                num_phase_bins // 2,
            )

            ends = starts + window_bins

            in_flux = (
                cumulative_flux[ends]
                - cumulative_flux[starts]
            )

            in_counts = (
                cumulative_counts[ends]
                - cumulative_counts[starts]
            )

            outside_flux = total_flux - in_flux
            outside_counts = total_points - in_counts

            valid = (
                (in_counts >= 8)
                & (outside_counts >= 20)
            )

            if not np.any(valid):
                continue

            transit_mean = np.full(
                num_phase_bins,
                np.nan,
            )
            outside_mean = np.full(
                num_phase_bins,
                np.nan,
            )

            transit_mean[valid] = (
                in_flux[valid]
                / in_counts[valid]
            )

            outside_mean[valid] = (
                outside_flux[valid]
                / outside_counts[valid]
            )

            depths = (
                outside_mean - transit_mean
            )

            effective_points = (
                in_counts
                * outside_counts
                / total_points
            )

            window_scores = (
                depths
                / robust_scatter
                * np.sqrt(
                    np.maximum(
                        effective_points,
                        0.0,
                    )
                )
            )

            window_scores[~valid] = -np.inf
            window_scores[depths <= 0] = -np.inf

            best_window_index = int(
                np.argmax(window_scores)
            )

            candidate_score = float(
                window_scores[best_window_index]
            )

            if candidate_score > local_best_score:

                local_best_score = candidate_score
                local_best_duration = duration_hours

                center_bin = (
                    best_window_index
                    + window_bins / 2.0
                )

                local_best_phase = (
                    center_bin
                    / num_phase_bins
                    - 0.5
                )

                if local_best_phase >= 0.5:
                    local_best_phase -= 1.0

        scores[period_index] = local_best_score

        if local_best_score > best_score:
            best_score = local_best_score
            best_period = period
            best_duration_hours = local_best_duration
            best_phase = local_best_phase

    if not np.isfinite(best_score):
        best_score = np.nan

    return {
        "best_period": float(best_period),
        "best_score": float(best_score),
        "best_duration_hours": float(best_duration_hours),
        "best_phase": float(best_phase),
        "periods": periods,
        "scores": scores,
    }


# ============================================================
# BLS period search
# ============================================================

def find_bls_period(
    time: np.ndarray,
    flux: np.ndarray,
) -> dict:
    """
    Two-stage BLS search.

    Stage 1:
        Search the complete 0.5-14 day range coarsely.

    Stage 2:
        Refine the best coarse period.

    This avoids using the archive period or a period estimated
    from candidate-event spacing.
    """

    # --------------------------------------------------------
    # Stage 1: broad coarse search
    # --------------------------------------------------------

    coarse = search_candidate_period(
        time=time,
        flux=flux,
        min_period=MIN_PERIOD_DAYS,
        max_period=MAX_PERIOD_DAYS,
        period_step=COARSE_PERIOD_STEP,
        num_phase_bins=COARSE_PHASE_BINS,
        duration_hours_grid=np.arange(
            0.5,
            8.01,
            1.0,
        ),
    )

    coarse_period = coarse["best_period"]

    if not np.isfinite(coarse_period):
        return {
            "period_days": np.nan,
            "score": np.nan,
            "duration_hours": np.nan,
            "phase": np.nan,
        }

    # --------------------------------------------------------
    # Stage 2: fine search around coarse solution
    # --------------------------------------------------------

    fine_min = max(
        MIN_PERIOD_DAYS,
        coarse_period - FINE_PERIOD_HALF_WIDTH,
    )

    fine_max = min(
        MAX_PERIOD_DAYS,
        coarse_period + FINE_PERIOD_HALF_WIDTH,
    )

    fine = search_candidate_period(
        time=time,
        flux=flux,
        min_period=fine_min,
        max_period=fine_max,
        period_step=FINE_PERIOD_STEP,
        num_phase_bins=FINE_PHASE_BINS,
        duration_hours_grid=np.arange(
            0.5,
            8.01,
            0.5,
        ),
    )

    return {
        "period_days": fine["best_period"],
        "score": fine["best_score"],
        "duration_hours": fine[
            "best_duration_hours"
        ],
        "phase": fine["best_phase"],
    }


# ============================================================
# Event support for BLS period
# ============================================================

def measure_period_support(
    event_measurements: list[dict],
    period: float,
    transit_duration_hours: float | None = None,
) -> tuple[int, float]:
    """
    Validate the BLS period using independently detected events.

    The timing tolerance is tied to the measured event duration,
    rather than allowing a large fraction of the orbital period.
    """

    if not np.isfinite(period) or period <= 0:
        return 0, np.nan

    strong_events = [
        event
        for event in event_measurements
        if event["num_points"] >= STRONG_EVENT_MIN_POINTS
    ]

    if len(strong_events) == 0:
        return 0, np.nan

    midpoints = np.array(
        [event["midpoint"] for event in strong_events],
        dtype=float,
    )

    reference_time = midpoints[0]

    epochs = np.rint(
        (midpoints - reference_time) / period
    )

    predicted_midpoints = (
        reference_time + epochs * period
    )

    residuals = (
        midpoints - predicted_midpoints
    )

    if (
        transit_duration_hours is not None
        and np.isfinite(transit_duration_hours)
        and transit_duration_hours > 0
    ):
        tolerance = max(
            0.25 * transit_duration_hours / 24.0,
            0.02,
        )
    else:
        tolerance = min(
            0.05 * period,
            0.05,
        )

    supported = (
        np.abs(residuals) <= tolerance
    )

    supporting_residuals = residuals[supported]
    num_supporting_events = len(
        supporting_residuals
    )

    if num_supporting_events < 2:
        return num_supporting_events, np.nan

    return (
        num_supporting_events,
        float(np.std(supporting_residuals)),
    )


# ============================================================
# Feature extraction
# ============================================================

def extract_features(
    time: np.ndarray,
    flux: np.ndarray,
    candidate_threshold: float = CANDIDATE_THRESHOLD,
    event_gap_minutes: float = EVENT_GAP_MINUTES,
) -> dict:
    """
    Extract the ML features for one preprocessed light curve.

    Period is determined by BLS.

    Candidate-event measurements are used for:
        - depth
        - duration
        - number of transit-like points
        - SNR
        - independent period support

    Archive parameters are NOT used.
    """

    finite = (
        np.isfinite(time)
        & np.isfinite(flux)
    )

    time = time[finite]
    flux = flux[finite]

    if len(time) < 100:
        return {
            "period_days": np.nan,
            "transit_depth": np.nan,
            "duration_hours": np.nan,
            "num_transit_points": np.nan,
            "baseline_scatter": np.nan,
            "transit_snr": np.nan,
            "num_events": 0.0,
            "period_scatter_days": np.nan,
            "bls_score": np.nan,
            "bls_duration_hours": np.nan,
            "bls_phase": np.nan,
        }

    # --------------------------------------------------------
    # Candidate event detection
    # --------------------------------------------------------

    candidate_mask = (
        flux < candidate_threshold
    )

    event_gap_days = (
        event_gap_minutes
        / (24.0 * 60.0)
    )

    events = group_candidate_points(
        time,
        candidate_mask,
        event_gap_days,
    )

    event_measurements = (
        measure_candidate_events(
            time,
            flux,
            events,
        )
    )

    strong_events = [
        event
        for event in event_measurements
        if event["num_points"]
        >= STRONG_EVENT_MIN_POINTS
    ]

    # --------------------------------------------------------
    # Event-dependent features
    # --------------------------------------------------------

    if len(strong_events) > 0:

        strongest_event = max(
            strong_events,
            key=lambda event: event["num_points"],
        )

        transit_depth = (
            strongest_event["depth"]
        )

        duration_hours = (
            strongest_event["duration_hours"]
        )

        num_transit_points = (
            strongest_event["num_points"]
        )

        strongest_start = (
            strongest_event["start_index"]
        )

        strongest_end = (
            strongest_event["end_index"]
        )

        outside_mask = np.ones(
            len(flux),
            dtype=bool,
        )

        outside_mask[
            strongest_start:
            strongest_end + 1
        ] = False

        outside_flux = flux[outside_mask]

        baseline_scatter = _robust_scatter(
            outside_flux
        )

        if (
            np.isfinite(baseline_scatter)
            and baseline_scatter > 0
        ):
            transit_snr = (
                transit_depth
                / baseline_scatter
            )
        else:
            transit_snr = np.nan

    else:

        transit_depth = np.nan
        duration_hours = np.nan
        num_transit_points = np.nan
        baseline_scatter = np.nan
        transit_snr = np.nan

    # --------------------------------------------------------
    # BLS period
    # --------------------------------------------------------

    bls = find_bls_period(
        time,
        flux,
    )

    period_days = bls["period_days"]

    # --------------------------------------------------------
    # Independent event support
    # --------------------------------------------------------

    num_events, period_scatter_days = (
        measure_period_support(
            event_measurements,
            period_days,
            duration_hours,
        )
    )

    return {
        "period_days": period_days,
        "transit_depth": transit_depth,
        "duration_hours": duration_hours,
        "num_transit_points": num_transit_points,
        "baseline_scatter": baseline_scatter,
        "transit_snr": transit_snr,
        "num_events": float(num_events),
        "period_scatter_days": period_scatter_days,

        # Diagnostic BLS features.
        # These are returned for analysis but are not yet
        # included in the original 8-feature vector.
        "bls_score": bls["score"],
        "bls_duration_hours": bls[
            "duration_hours"
        ],
        "bls_phase": bls["phase"],
    }


# ============================================================
# Feature vector
# ============================================================

def create_feature_vector(
    features: dict,
) -> np.ndarray:
    """
    Convert extracted features into the original
    8-dimensional ML feature vector.

    Order:
        1. period_days
        2. transit_depth
        3. duration_hours
        4. num_transit_points
        5. baseline_scatter
        6. transit_snr
        7. num_events
        8. period_scatter_days
    """

    return np.array(
        [
            features["period_days"],
            features["transit_depth"],
            features["duration_hours"],
            features["num_transit_points"],
            features["baseline_scatter"],
            features["transit_snr"],
            features["num_events"],
            features["period_scatter_days"],
        ],
        dtype=float,
    )


# ============================================================
# FITS preprocessing
# ============================================================

def load_lightcurve(
    data_file: Path,
):
    """
    Load the TESS SPOC light curve.
    """

    from astropy.io import fits

    with fits.open(data_file) as hdul:

        data = hdul[1].data

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

    return (
        time,
        flux,
        flux_err,
        quality,
    )


def filter_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    quality: np.ndarray,
):
    """
    Remove invalid and quality-flagged measurements.
    """

    finite_mask = (
        np.isfinite(time)
        & np.isfinite(flux)
    )

    quality_mask = (
        quality == 0
    )

    mask = (
        finite_mask
        & quality_mask
    )

    return (
        time[mask],
        flux[mask],
        flux_err[mask],
    )


def normalize_flux(
    flux: np.ndarray,
):
    """
    Normalize flux by its median.
    """

    median_flux = np.median(flux)

    if (
        not np.isfinite(median_flux)
        or median_flux == 0
    ):
        raise ValueError(
            "Invalid median flux during normalization."
        )

    return (
        flux / median_flux,
        median_flux,
    )


def detrend_flux(
    time: np.ndarray,
    normalized_flux: np.ndarray,
    cadence_minutes: float,
    window_minutes: float = 24.0 * 60.0,
):
    """
    Remove long-term trends using a median filter.

    Keeps the Day 2 1-day detrending strategy.
    """

    from scipy.ndimage import median_filter

    window_points = int(
        window_minutes / cadence_minutes
    )

    # Median filter requires an odd window.
    if window_points % 2 == 0:
        window_points += 1

    window_points = max(
        window_points,
        3,
    )

    baseline = median_filter(
        normalized_flux,
        size=window_points,
        mode="nearest",
    )

    baseline_median = np.median(
        baseline
    )

    if (
        not np.isfinite(baseline_median)
        or baseline_median == 0
    ):
        raise ValueError(
            "Invalid baseline during detrending."
        )

    detrended_flux = (
        normalized_flux / baseline
    ) * baseline_median

    return (
        detrended_flux,
        baseline,
        window_points,
    )


def preprocess_lightcurve(
    data_file: Path,
):
    """
    Complete preprocessing pipeline.

    Returns
    -------
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
    ) = load_lightcurve(
        data_file
    )

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

    if len(clean_time) < 100:
        raise ValueError(
            "Too few valid measurements."
        )

    # --------------------------------------------------------
    # Cadence
    # --------------------------------------------------------

    time_differences = np.diff(
        clean_time
    )

    valid_differences = (
        time_differences[
            np.isfinite(time_differences)
            & (time_differences > 0)
        ]
    )

    if len(valid_differences) == 0:
        raise ValueError(
            "Could not determine cadence."
        )

    cadence_minutes = (
        np.median(valid_differences)
        * 24.0
        * 60.0
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized_flux, _ = (
        normalize_flux(
            clean_flux
        )
    )

    # --------------------------------------------------------
    # Detrend
    # --------------------------------------------------------

    (
        detrended_flux,
        baseline,
        window_points,
    ) = detrend_flux(
        clean_time,
        normalized_flux,
        cadence_minutes,
    )

    return (
        clean_time,
        detrended_flux,
        clean_flux_err,
        baseline,
        cadence_minutes,
        window_points,
    )


# ============================================================
# Command-line interface
# ============================================================

def main():
    """
    Run feature extraction for one FITS file.
    """

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Extract AstroHunter light-curve "
            "features using BLS period search."
        )
    )

    parser.add_argument(
        "data_file",
        type=Path,
        help="Path to TESS light-curve FITS file.",
    )

    args = parser.parse_args()

    print(
        "\n=== Preprocessing ==="
    )

    (
        time,
        detrended_flux,
        flux_err,
        baseline,
        cadence_minutes,
        window_points,
    ) = preprocess_lightcurve(
        args.data_file
    )

    print(
        f"Cadence:          "
        f"{cadence_minutes:.2f} minutes"
    )

    print(
        f"Detrend window:   "
        f"{window_points} points"
    )

    print(
        "\n=== BLS Period Search ==="
    )

    features = extract_features(
        time,
        detrended_flux,
    )

    print(
        f"BLS period:       "
        f"{features['period_days']:.6f} days"
    )

    print(
        f"BLS score:        "
        f"{features['bls_score']:.4f}"
    )

    print(
        f"BLS duration:     "
        f"{features['bls_duration_hours']:.2f} hours"
    )

    print(
        f"BLS phase:        "
        f"{features['bls_phase']:.6f}"
    )

    print(
        "\n=== Event Features ==="
    )

    print(
        f"Transit depth:    "
        f"{features['transit_depth']:.3%}"
    )

    print(
        f"Duration:          "
        f"{features['duration_hours']:.2f} hours"
    )

    print(
        f"Transit points:    "
        f"{features['num_transit_points']:.0f}"
    )

    print(
        f"Baseline scatter:  "
        f"{features['baseline_scatter']:.6f}"
    )

    print(
        f"Transit SNR:       "
        f"{features['transit_snr']:.2f}"
    )

    print(
        f"Supporting events: "
        f"{features['num_events']:.0f}"
    )

    print(
        f"Period scatter:    "
        f"{features['period_scatter_days']:.6f} days"
    )

    print(
        "\n=== Feature Vector ==="
    )

    feature_vector = create_feature_vector(
        features
    )

    names = [
        "period_days",
        "transit_depth",
        "duration_hours",
        "num_transit_points",
        "baseline_scatter",
        "transit_snr",
        "num_events",
        "period_scatter_days",
    ]

    for name, value in zip(
        names,
        feature_vector,
    ):
        print(
            f"{name:<24}: "
            f"{value:.6f}"
        )


if __name__ == "__main__":
    main()