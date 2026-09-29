from pathlib import Path

import numpy as np

from astro_hunter.preprocess import preprocess_lightcurve
from astro_hunter.feature_extraction import (
    group_candidate_points,
    measure_candidate_events,
)


DATA_FILE = Path(
    "data/raw/mastDownload/TESS/"
    "tess2021175071901-s0040-0000000394050135-0211-s/"
    "tess2021175071901-s0040-0000000394050135-0211-s_lc.fits"
)


def main():
    time, flux, flux_err, baseline, cadence_minutes, window_points = (
        preprocess_lightcurve(DATA_FILE)
    )

    threshold = 0.997

    candidate_mask = flux < threshold

    # TESS time is measured in days.
    # 10 minutes = 10 / (24 * 60) days.
    max_gap_days = 10.0 / (24.0 * 60.0)

    events = group_candidate_points(
        time,
        candidate_mask,
        max_gap_days,
    )

    measurements = measure_candidate_events(
        time,
        flux,
        events,
    )

    print("\n=== Strong Event Analysis ===")
    print(f"Total raw candidate events: {len(measurements)}")

    if not measurements:
        print("No candidate events detected.")
        return

    # Ignore tiny threshold crossings.
    # We want events containing enough consecutive cadences
    # to plausibly represent an astronomical dip.
    min_points = 20

    strong_events = [
        event
        for event in measurements
        if event["num_points"] >= min_points
    ]

    strong_events.sort(key=lambda event: event["midpoint"])

    print(f"Strong events (>= {min_points} points): {len(strong_events)}")

    print("\n=== Strong Events ===")

    for i, event in enumerate(strong_events):
        duration_hours = event["duration"] * 24.0

        print(
            f"{i}: "
            f"midpoint={event['midpoint']:.8f}, "
            f"duration={duration_hours:.3f} h, "
            f"depth={event['depth']:.6f}, "
            f"points={event['num_points']}"
        )

    if len(strong_events) < 2:
        print("\nNot enough strong events to estimate a period.")
        print("\n=== Done ===")
        return

    # ---------------------------------------------------------
    # Compare the midpoints of strong events
    # ---------------------------------------------------------

    midpoints = np.array(
        [event["midpoint"] for event in strong_events],
        dtype=float,
    )

    print("\n=== Strong Event Separations ===")

    differences = np.diff(midpoints)

    for i, diff in enumerate(differences):
        print(
            f"{i} -> {i + 1}: "
            f"{diff:.8f} days "
            f"({diff * 24.0:.3f} hours)"
        )

    print(
        f"\nMedian strong-event separation: "
        f"{np.median(differences):.8f} days"
    )

    print(
        f"Median strong-event separation: "
        f"{np.median(differences) * 24.0:.3f} hours"
    )

    print("\n=== Done ===")


if __name__ == "__main__":
    main()