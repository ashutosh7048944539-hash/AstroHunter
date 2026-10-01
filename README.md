# AstroHunter

AstroHunter is an astronomical event-detection pipeline for identifying candidate exoplanet transits from real stellar light-curve data.

The project combines astronomical data retrieval, signal processing, statistical period detection, feature extraction, and machine-learning analysis to turn raw TESS observations into validated transit candidates.

## Pipeline

```text
TESS light curves
        |
        v
Data quality filtering
        |
        v
Flux normalization
        |
        v
Long-term trend removal
        |
        v
Transit/event detection
        |
        v
Feature extraction
        |
        v
BLS period search
        |
        v
Candidate validation
        |
        v
Machine-learning analysis
```

## Current Result

AstroHunter has been tested on a real TESS light curve for **TOI-2025 / TIC 394050135** from Sector 40.

The validation pipeline detected:

- Period: **8.8728 days**
- Transit depth: **~0.91%**
- Transit duration: **~3.3 hours**
- Supporting transit events: **4**
- Transit SNR: **5.21**
- BLS score: **58.71**

The resulting phase-folded validation plot is available at:

`reports/toi_2025_validation.png`

## Project Structure



## Installation

Create a virtual environment if you do not already have one:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Pipeline

From the project root:

```powershell
$env:PYTHONPATH="src"
```

Run preprocessing:

```powershell
python -m astro_hunter.preprocess
```

Run candidate validation:

```powershell
python -m astro_hunter.validate_candidate
```

## Data

AstroHunter works with light curves obtained from NASA's Mikulski Archive for Space Telescopes (MAST).

Large raw datasets are intentionally excluded from the Git repository. The repository contains the code and configuration required to reproduce the analysis after obtaining the required observational data.

## Reproducibility

The Python dependencies used by the project are pinned in `requirements.txt`.

The analysis pipeline is implemented as Python modules under `src/astro_hunter/`, allowing the processing and validation steps to be reproduced from the command line.

## Status

The core light-curve processing, transit detection, feature extraction, BLS period search, and candidate validation pipeline is implemented.

Current work focuses on improving reproducibility, documentation, candidate analysis, and machine-learning evaluation.