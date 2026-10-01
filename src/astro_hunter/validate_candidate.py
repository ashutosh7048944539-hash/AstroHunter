from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from astro_hunter.preprocess import preprocess_lightcurve
from astro_hunter.feature_extraction import (
    find_bls_period,
    phase_fold_lightcurve,
    bin_phase_folded_lightcurve,
    extract_features,
)


DATA_FILE = Path(
    "data/raw/mastDownload/TESS/"
    "tess2021175071901-s0040-0000000394050135-0211-s/"
    "tess2021175071901-s0040-0000000394050135-0211-s_lc.fits"
)

REPORTS_DIR = Path("reports")

def main():

    print("=" * 60)
    print("AstroHunter — Candidate Validation")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Preprocess light curve
    # ---------------------------------------------------------

    (
        time,
        flux,
        flux_err,
        baseline,
        cadence_minutes,
        window_points,
    ) = preprocess_lightcurve(DATA_FILE)

    print("\nPreprocessing complete.")
    print(f"Usable measurements : {len(time)}")
    print(f"Cadence             : {cadence_minutes:.2f} min")
    print(f"Detrending window   : {window_points} points")

    # ---------------------------------------------------------
    # 2. BLS period search
    # ---------------------------------------------------------

    bls_result = find_bls_period(
        time=time,
        flux=flux,
    )

    period = bls_result["period_days"]
    score = bls_result["score"]
    bls_duration = bls_result["duration_hours"]
    bls_phase = bls_result["phase"]

    print("\nBLS result:")
    print(f"Detected period     : {period:.6f} days")
    print(f"BLS score           : {score:.4f}")
    print(f"Transit duration    : {bls_duration:.3f} hours")
    print(f"BLS phase           : {bls_phase:.4f}")

    # ---------------------------------------------------------
    # 3. Phase folding
    # ---------------------------------------------------------

    # Convert the BLS phase convention to the phase convention
    # used by the phase-folded visualization.

    transit_phase = bls_phase + 0.5

    if transit_phase >= 0.5:
        transit_phase -= 1.0

    phase, folded_flux = phase_fold_lightcurve(
        time=time,
        flux=flux,
        period=period,
        reference_phase=transit_phase,
    )

    print("\nPhase folding complete.")
    print(
        f"Phase range         : "
        f"{phase.min():.4f} to {phase.max():.4f}"
    )
    print(f"Folded points       : {len(phase)}")

    # ---------------------------------------------------------
    # 4. Extract scientific features
    # ---------------------------------------------------------

    features = extract_features(
        time=time,
        flux=flux,
    )

    transit_depth = features["transit_depth"]
    duration_hours = features["duration_hours"]
    num_events = int(features["num_events"])
    transit_snr = features["transit_snr"]
    period_scatter = features["period_scatter_days"]

    print("\nScientific validation:")
    print(
        f"Transit depth       : "
        f"{transit_depth * 100:.3f}%"
    )
    print(
        f"Event duration      : "
        f"{duration_hours:.2f} hours"
    )
    print(f"Supporting events   : {num_events}")
    print(f"Transit SNR         : {transit_snr:.2f}")
    print(
        f"Period scatter      : "
        f"{period_scatter:.6f} days"
    )

    # ---------------------------------------------------------
    # 5. Bin phase-folded light curve
    # ---------------------------------------------------------

    (
        binned_phase,
        binned_flux,
        bin_counts,
    ) = bin_phase_folded_lightcurve(
        phase=phase,
        flux=folded_flux,
        num_bins=200,
    )

    # ---------------------------------------------------------
    # 6. Create scientific validation plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    # Raw TESS measurements
    ax.scatter(
        phase,
        folded_flux,
        s=3,
        alpha=0.20,
        label="TESS measurements",
    )

    # Binned light curve
    valid = (
        np.isfinite(binned_phase)
        & np.isfinite(binned_flux)
        & (bin_counts > 0)
    )

    ax.plot(
        binned_phase[valid],
        binned_flux[valid],
        linewidth=2,
        label="Binned light curve",
    )

    # Transit center
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="Transit center",
    )

    # Normalized baseline
    ax.axhline(
        1.0,
        linestyle=":",
        linewidth=1,
        alpha=0.7,
    )

    # ---------------------------------------------------------
    # 7. Scientific annotation
    # ---------------------------------------------------------

    annotation = (
        f"Period = {period:.4f} d\n"
        f"Depth = {transit_depth * 100:.2f}%\n"
        f"Duration = {duration_hours:.2f} h\n"
        f"Events = {num_events}\n"
        f"Transit SNR = {transit_snr:.2f}\n"
        f"Period scatter = {period_scatter:.5f} d"
    )

    ax.text(
        0.02,
        0.05,
        annotation,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            alpha=0.85,
        ),
    )

    # ---------------------------------------------------------
    # 8. Plot formatting
    # ---------------------------------------------------------

    ax.set_xlabel(
        "Orbital Phase"
    )

    ax.set_ylabel(
        "Normalized Flux"
    )

    ax.set_title(
        "TOI-2025 — Phase-Folded Transit Validation"
        f"\nP = {period:.4f} days"
    )

    ax.set_xlim(
        -0.5,
        0.5,
    )

    ax.grid(
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        REPORTS_DIR
        / "toi_2025_validation.png"
    )

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"\nValidation figure saved to: "
        f"{output_file}"
    )

    plt.show()

    # ---------------------------------------------------------
    # 9. Completion
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "Candidate validation completed."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()