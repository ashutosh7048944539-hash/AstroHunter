from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter


DATA_FILE = Path(
    "data/raw/mastDownload/TESS/"
    "tess2021175071901-s0040-0000000394050135-0211-s/"
    "tess2021175071901-s0040-0000000394050135-0211-s_lc.fits"
)


def load_preprocessed_lightcurve():
    """Load, quality-filter, normalize, and detrend the TESS light curve."""

    with fits.open(DATA_FILE) as hdul:
        data = hdul["LIGHTCURVE"].data

        time = np.asarray(data["TIME"], dtype=float)
        flux = np.asarray(data["PDCSAP_FLUX"], dtype=float)
        quality = np.asarray(data["QUALITY"])

    # ---------------------------------------------------------
    # Remove invalid measurements and flagged cadences.
    # ---------------------------------------------------------

    finite_mask = np.isfinite(time) & np.isfinite(flux)
    quality_mask = quality == 0
    combined_mask = finite_mask & quality_mask

    clean_time = time[combined_mask]
    clean_flux = flux[combined_mask]

    # ---------------------------------------------------------
    # Normalize the flux.
    # ---------------------------------------------------------

    baseline = np.median(clean_flux)
    normalized_flux = clean_flux / baseline

    # ---------------------------------------------------------
    # Estimate cadence directly from the observations.
    # ---------------------------------------------------------

    time_differences = np.diff(clean_time)

    cadence_days = np.median(time_differences)
    cadence_minutes = cadence_days * 24 * 60

    # ---------------------------------------------------------
    # Detrend using the same 1-day median-filter window
    # used during Day 2.
    # ---------------------------------------------------------

    window_minutes = 24 * 60

    window_points = int(
        window_minutes / cadence_minutes
    )

    if window_points % 2 == 0:
        window_points += 1

    window_points = max(window_points, 3)

    trend = median_filter(
        normalized_flux,
        size=window_points,
        mode="nearest",
    )

    detrended_flux = normalized_flux / trend

    return (
        clean_time,
        detrended_flux,
        cadence_minutes,
    )


def group_candidate_points(
    time,
    candidate_mask,
    max_gap,
):
    """
    Group nearby low-flux candidate points into events.

    A short gap above the threshold does not immediately
    split the event. A new event starts only when the time
    since the previous candidate point exceeds max_gap.
    """

    events = []

    start_index = None
    last_candidate_index = None

    for i in range(len(time)):

        if not candidate_mask[i]:
            continue

        # Start a new event.
        if start_index is None:
            start_index = i
            last_candidate_index = i
            continue

        candidate_gap = (
            time[i] - time[last_candidate_index]
        )

        # Start a new event if the gap is too large.
        if candidate_gap > max_gap:

            events.append(
                (
                    start_index,
                    last_candidate_index,
                )
            )

            start_index = i

        last_candidate_index = i

    # Close the final event.
    if start_index is not None:

        events.append(
            (
                start_index,
                last_candidate_index,
            )
        )

    return events


def measure_candidate_events(
    time,
    flux,
    events,
):
    """Measure basic properties of candidate events."""

    measurements = []

    for start_index, end_index in events:

        event_time = time[
            start_index:end_index + 1
        ]

        event_flux = flux[
            start_index:end_index + 1
        ]

        duration = (
            event_time[-1]
            - event_time[0]
        )

        minimum_flux = np.min(event_flux)

        depth = 1.0 - minimum_flux

        midpoint = (
            event_time[0]
            + event_time[-1]
        ) / 2

        measurements.append(
            {
                "start_index": start_index,
                "end_index": end_index,
                "midpoint": midpoint,
                "duration": duration,
                "minimum_flux": minimum_flux,
                "depth": depth,
                "num_points": len(event_time),
            }
        )

    return measurements


def estimate_period_from_events(events):
    """Estimate a rough period from separated strong events."""

    if len(events) < 2:
        return None

    sorted_events = sorted(
        events,
        key=lambda event: event["midpoint"],
    )

    midpoints = np.array(
        [
            event["midpoint"]
            for event in sorted_events
        ],
        dtype=float,
    )

    periods = np.diff(midpoints)

    return periods


def bin_phase_folded_lightcurve(
    phase,
    flux,
    num_bins=100,
):
    """Bin a phase-folded light curve using median flux."""

    bin_edges = np.linspace(
        -0.5,
        0.5,
        num_bins + 1,
    )

    bin_centers = (
        bin_edges[:-1]
        + bin_edges[1:]
    ) / 2

    binned_flux = np.full(
        num_bins,
        np.nan,
    )

    binned_scatter = np.full(
        num_bins,
        np.nan,
    )

    points_per_bin = np.zeros(
        num_bins,
        dtype=int,
    )

    bin_indices = np.digitize(
        phase,
        bin_edges,
    ) - 1

    for i in range(num_bins):

        mask = bin_indices == i

        if not np.any(mask):
            continue

        bin_flux = flux[mask]

        binned_flux[i] = np.median(
            bin_flux
        )

        binned_scatter[i] = np.std(
            bin_flux
        )

        points_per_bin[i] = len(
            bin_flux
        )

    valid = np.isfinite(
        binned_flux
    )

    return (
        bin_centers[valid],
        binned_flux[valid],
        binned_scatter[valid],
        points_per_bin[valid],
    )


def phase_fold_lightcurve(
    time,
    flux,
    period,
    reference_time,
):
    """
    Fold the light curve so reference_time
    corresponds to phase zero.
    """

    phase = (
        (time - reference_time)
        / period
        + 0.5
    ) % 1.0 - 0.5

    return phase, flux


def search_candidate_period(
    time,
    flux,
    min_period,
    max_period,
    period_step=0.0002,
    num_phase_bins=400,
):
    """
    Search for the period producing the strongest
    repeated transit-like signal.

    This is a simplified Box Least Squares (BLS)-style
    search.

    The score combines:

        transit depth
        -------------------------
        out-of-transit scatter

    multiplied by sqrt(number of in-transit points).

    This is an exploratory detection statistic,
    not a formal statistical significance.
    """

    periods = np.arange(
        min_period,
        max_period + period_step / 2,
        period_step,
    )

    # Plausible transit durations.
    duration_hours = np.array(
        [
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
        ],
        dtype=float,
    )

    scores = np.full(
        len(periods),
        -np.inf,
    )

    best_durations = np.full(
        len(periods),
        np.nan,
    )

    best_phases = np.full(
        len(periods),
        np.nan,
    )

    reference_time = time[0]

    bin_edges = np.linspace(
        -0.5,
        0.5,
        num_phase_bins + 1,
    )

    phase_centers = (
        bin_edges[:-1]
        + bin_edges[1:]
    ) / 2

    # ---------------------------------------------------------
    # Robust estimate of the overall scatter.
    # ---------------------------------------------------------

    median_flux = np.median(flux)

    residuals = flux - median_flux

    robust_scatter = (
        1.4826
        * np.median(
            np.abs(
                residuals
                - np.median(residuals)
            )
        )
    )

    robust_scatter = max(
        robust_scatter,
        1e-6,
    )

    # ---------------------------------------------------------
    # Search trial periods.
    # ---------------------------------------------------------

    for period_index, period in enumerate(
        periods
    ):

        # Fold the light curve.
        phase = (
            (time - reference_time)
            / period
            + 0.5
        ) % 1.0 - 0.5

        bin_indices = np.digitize(
            phase,
            bin_edges,
        ) - 1

        valid = (
            (bin_indices >= 0)
            & (bin_indices < num_phase_bins)
        )

        counts = np.bincount(
            bin_indices[valid],
            minlength=num_phase_bins,
        )

        flux_sums = np.bincount(
            bin_indices[valid],
            weights=flux[valid],
            minlength=num_phase_bins,
        )

        valid_bins = counts > 0

        # -----------------------------------------------------
        # Try different transit durations.
        # -----------------------------------------------------

        for hours in duration_hours:

            duration_days = (
                hours / 24.0
            )

            half_width = (
                duration_days
                / (2.0 * period)
            )

            # -------------------------------------------------
            # Try every possible transit phase.
            # -------------------------------------------------

            for center_index in range(
                num_phase_bins
            ):

                distance = np.abs(
                    phase_centers
                    - phase_centers[
                        center_index
                    ]
                )

                # Phase is circular.
                distance = np.minimum(
                    distance,
                    1.0 - distance,
                )

                inside = (
                    distance <= half_width
                )

                if not np.any(inside):
                    continue

                transit_counts = (
                    counts[inside]
                )

                total_points = (
                    transit_counts.sum()
                )

                # Need enough measurements.
                if total_points < 20:
                    continue

                transit_flux_sum = (
                    flux_sums[inside].sum()
                )

                transit_mean = (
                    transit_flux_sum
                    / total_points
                )

                # -------------------------------------------------
                # Estimate the out-of-transit baseline.
                # -------------------------------------------------

                outside = (
                    valid_bins
                    & ~inside
                )

                if not np.any(outside):
                    continue

                outside_counts = (
                    counts[outside]
                )

                outside_flux_sum = (
                    flux_sums[outside].sum()
                )

                outside_points = (
                    outside_counts.sum()
                )

                if outside_points < 100:
                    continue

                outside_mean = (
                    outside_flux_sum
                    / outside_points
                )

                # -------------------------------------------------
                # Transit depth.
                # -------------------------------------------------

                depth = (
                    outside_mean
                    - transit_mean
                )

                # Ignore brightening events.
                if depth <= 0:
                    continue

                # -------------------------------------------------
                # BLS-style detection statistic.
                # -------------------------------------------------

                score = (
                    depth
                    / robust_scatter
                    * np.sqrt(total_points)
                )

                if score > scores[
                    period_index
                ]:

                    scores[
                        period_index
                    ] = score

                    best_durations[
                        period_index
                    ] = hours

                    best_phases[
                        period_index
                    ] = phase_centers[
                        center_index
                    ]

    best_index = np.argmax(
        scores
    )

    return (
        periods,
        scores,
        best_durations,
        best_phases,
        best_index,
    )


def main() -> None:

    # =========================================================
    # STEP 0: PREPROCESS
    # =========================================================

    (
        time,
        flux,
        cadence_minutes,
    ) = load_preprocessed_lightcurve()

    print()
    print(
        "=== Day 3: Preprocessed light curve ==="
    )

    print(
        f"Measurements:            "
        f"{len(time):,}"
    )

    print(
        f"Cadence:                 "
        f"{cadence_minutes:.2f} minutes"
    )

    print(
        f"Median flux:             "
        f"{np.median(flux):.6f}"
    )

    print(
        f"Minimum flux:            "
        f"{flux.min():.6f}"
    )

    print(
        f"Maximum flux:            "
        f"{flux.max():.6f}"
    )

    # =========================================================
    # STEP 1: THRESHOLD-BASED CANDIDATE POINTS
    # =========================================================

    threshold = 0.997

    candidate_mask = (
        flux < threshold
    )

    candidate_time = time[
        candidate_mask
    ]

    candidate_flux = flux[
        candidate_mask
    ]

    print()
    print(
        "=== Candidate transit points ==="
    )

    print(
        f"Threshold:               "
        f"{threshold:.3f}"
    )

    print(
        f"Candidate points:        "
        f"{candidate_mask.sum():,}"
    )

    # =========================================================
    # STEP 2: GROUP CANDIDATE POINTS
    # =========================================================

    event_gap_minutes = 10.0

    max_gap = (
        event_gap_minutes
        / (24 * 60)
    )

    events = group_candidate_points(
        time,
        candidate_mask,
        max_gap,
    )

    event_measurements = (
        measure_candidate_events(
            time,
            flux,
            events,
        )
    )

    print()
    print(
        "=== Candidate events ==="
    )

    print(
        f"Maximum event gap:       "
        f"{event_gap_minutes:.2f} minutes"
    )

    print(
        f"Candidate events:        "
        f"{len(event_measurements):,}"
    )

    # =========================================================
    # STEP 3: EVENT STATISTICS
    # =========================================================

    if event_measurements:

        event_sizes = np.array(
            [
                event["num_points"]
                for event in event_measurements
            ]
        )

        event_durations = np.array(
            [
                event["duration"]
                for event in event_measurements
            ]
        )

        print()
        print(
            "=== Candidate event statistics ==="
        )

        print(
            f"Total events:            "
            f"{len(event_measurements):,}"
        )

        print(
            f"Smallest event:          "
            f"{event_sizes.min()} points"
        )

        print(
            f"Largest event:           "
            f"{event_sizes.max()} points"
        )

        print(
            f"Median event size:       "
            f"{np.median(event_sizes):.0f} points"
        )

        print(
            f"Events with >= 5 points: "
            f"{np.sum(event_sizes >= 5):,}"
        )

        print(
            f"Events with >= 10 points: "
            f"{np.sum(event_sizes >= 10):,}"
        )

        print(
            f"Longest duration:        "
            f"{event_durations.max() * 24 * 60:.2f} minutes"
        )

    # =========================================================
    # STEP 4: LONG CANDIDATE EVENTS
    # =========================================================

    long_events = [
        event
        for event in event_measurements
        if event["num_points"] >= 5
    ]

    long_events.sort(
        key=lambda event: event[
            "num_points"
        ],
        reverse=True,
    )

    print()
    print(
        "=== Longest candidate events ==="
    )

    print(
        "Midpoint (BJD)       Points    "
        "Duration (min)    Depth"
    )

    for event in long_events[:15]:

        print(
            f"{event['midpoint']:18.5f} "
            f"{event['num_points']:8d} "
            f"{event['duration'] * 24 * 60:16.2f} "
            f"{event['depth'] * 100:8.3f}%"
        )

    # =========================================================
    # STEP 5: ROUGH PERIOD FROM STRONG EVENTS
    # =========================================================

    strong_events = [
        event
        for event in event_measurements
        if event["num_points"] >= 50
    ]

    periods = estimate_period_from_events(
        strong_events
    )

    candidate_period = None

    print()
    print(
        "=== Candidate period estimate ==="
    )

    if periods is None:

        print(
            "Not enough strong events "
            "to estimate a period."
        )

    else:

        candidate_period = np.median(
            periods
        )

        print(
            f"Strong events:           "
            f"{len(strong_events):,}"
        )

        print(
            "Event-to-event periods:"
        )

        for period in periods:

            print(
                f"  {period:.5f} days"
            )

        print(
            f"Median candidate period: "
            f"{candidate_period:.5f} days"
        )

        print(
            f"Period scatter:          "
            f"{np.std(periods):.5f} days"
        )

    # =========================================================
    # STEP 6: REFINED PERIOD SEARCH
    # =========================================================

    if candidate_period is not None:

        search_min_period = (
            candidate_period - 0.10
        )

        search_max_period = (
            candidate_period + 0.10
        )

    else:

        search_min_period = 7.5
        search_max_period = 10.5

    (
        search_periods,
        period_scores,
        best_durations,
        best_phases,
        best_index,
    ) = search_candidate_period(
        time,
        flux,
        min_period=search_min_period,
        max_period=search_max_period,
        period_step=0.0002,
        num_phase_bins=400,
    )

    detected_period = (
        search_periods[best_index]
    )

    detected_phase = (
        best_phases[best_index]
    )

    print()
    print(
        "=== Period search ==="
    )

    print(
        f"Search range:           "
        f"{search_periods[0]:.4f} - "
        f"{search_periods[-1]:.4f} days"
    )

    print(
        f"Number of trial periods: "
        f"{len(search_periods):,}"
    )

    print(
        f"Best candidate period:  "
        f"{detected_period:.5f} days"
    )

    print(
        f"BLS-style score:        "
        f"{period_scores[best_index]:.5f}"
    )

    print(
        f"Best duration:          "
        f"{best_durations[best_index]:.2f} hours"
    )

    print(
        f"Best transit phase:     "
        f"{detected_phase:.4f}"
    )

    if candidate_period is not None:

        period_difference_minutes = (
            abs(
                detected_period
                - candidate_period
            )
            * 24
            * 60
        )

        print(
            f"Difference from event "
            f"estimate:              "
            f"{period_difference_minutes:.2f} minutes"
        )

    # =========================================================
    # STEP 7: PHASE FOLDING
    # =========================================================

    reference_time = (
        time[0]
        + detected_phase
        * detected_period
    )

    phase, folded_flux = (
        phase_fold_lightcurve(
            time,
            flux,
            detected_period,
            reference_time,
        )
    )

    (
        binned_phase,
        binned_flux,
        binned_scatter,
        points_per_bin,
    ) = bin_phase_folded_lightcurve(
        phase,
        folded_flux,
        num_bins=100,
    )

    print()
    print(
        "=== Phase folding ==="
    )

    print(
        f"Detected period:        "
        f"{detected_period:.5f} days"
    )

    print(
        f"Phase bins:             "
        f"{len(binned_phase):,}"
    )

    print(
        f"Median points/bin:      "
        f"{np.median(points_per_bin):.0f}"
    )

    print(
        f"Reference transit time: "
        f"{reference_time:.5f}"
    )

    # =========================================================
    # PLOT 1: PERIOD SEARCH
    # =========================================================

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        search_periods,
        period_scores,
        "-",
        linewidth=1,
    )

    plt.axvline(
        detected_period,
        linestyle="--",
        linewidth=1,
        label=(
            f"Detected period = "
            f"{detected_period:.4f} d"
        ),
    )

    if candidate_period is not None:

        plt.axvline(
            candidate_period,
            linestyle=":",
            linewidth=1,
            label=(
                f"Event estimate = "
                f"{candidate_period:.4f} d"
            ),
        )

    plt.xlabel(
        "Trial period (days)"
    )

    plt.ylabel(
        "BLS-style detection score"
    )

    plt.title(
        "AstroHunter Period Search"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

    # =========================================================
    # PLOT 2: PHASE-FOLDED LIGHT CURVE
    # =========================================================

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        phase,
        folded_flux,
        ".",
        markersize=1,
        alpha=0.25,
        label="Phase-folded light curve",
    )

    plt.plot(
        binned_phase,
        binned_flux,
        "o-",
        markersize=3,
        linewidth=1,
        label="Binned median flux",
    )

    plt.axvline(
        0.0,
        linestyle="--",
        linewidth=1,
        label="Transit phase zero",
    )

    plt.xlabel(
        "Phase"
    )

    plt.ylabel(
        "Detrended flux"
    )

    plt.title(
        "Phase-Folded TESS Light Curve — "
        "Detected Period"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

    # =========================================================
    # PLOT 3: ORIGINAL TIME-DOMAIN DETECTION
    # =========================================================

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time,
        flux,
        ".",
        markersize=1,
        label="Detrended light curve",
    )

    plt.plot(
        candidate_time,
        candidate_flux,
        ".",
        markersize=2,
        label="Candidate low-flux points",
    )

    plt.axhline(
        threshold,
        linestyle="--",
        linewidth=1,
        label="Detection threshold",
    )

    plt.xlabel(
        "Time (BJD - 2457000)"
    )

    plt.ylabel(
        "Detrended flux"
    )

    plt.title(
        "Initial Transit Candidate Detection"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()