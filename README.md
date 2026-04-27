# Exposure-Aligned Target Trial Emulation

<img width="9663" height="6733" alt="F1" src="https://github.com/user-attachments/assets/4b0e4b5f-7760-4fbc-b9e7-2b7769fecf28" />

This repository accompanies the simulation experiments for exposure-aligned target trial emulation of long-latency medication outcomes. It contains public simulation code, manuscript-ready figure builders, supplementary table builders, and the final 200-replicate simulation outputs.

## Overview

Initiation-based target trial emulation can be inadequate when treatment effects depend on post-baseline exposure history, including persistence, discontinuation, gaps, restarts, and accumulated exposure over time. This repository provides reproducible simulation code for evaluating exposure-aligned estimation against initiation-only, naive persistent-user, and cumulative exposure approaches.

## Repository Layout

```text
Exposure-Aligned-TTE/
  README.md
  data/
    simulated.py
  docs/
    figure1_framework.png
  src/
    exposure_aligned_tte/
      __init__.py
      plotting.py
      simulation_manuscript.py
      build_simulation_supplementary_tables.py
      build_simulation_figure3.py
  results/
    final_200rep/
      figures/
      tables/
      source_csv/
```

## Code

Core public code is under `src/exposure_aligned_tte/`.

- `simulation_manuscript.py`: simulation engine and manuscript figure builders
- `build_simulation_supplementary_tables.py`: builds Supplementary Tables S1 to S5
- `build_simulation_figure3.py`: builds multi-panel Figure 3
- `plotting.py`: shared plotting utilities

The file `data/simulated.py` provides a lightweight registry for bundled public simulation outputs.

## Bundled Results

Final 200-replicate outputs are included under `results/final_200rep/`.

- `figures/`: Figure 1, Figure 2, and Figure 3 in PNG and SVG formats
- `tables/`: Supplementary Tables S1 to S5
- `source_csv/`: source CSV files used to rebuild the bundled figures and tables

## Installation

Install the package in editable mode:

```bash
pip install -e .
```

Or create the conda environment:

```bash
conda env create -f environment.yml
conda activate exposure-aligned-tte
pip install -e .
```

## Rebuild Bundled Outputs

Rebuild Supplementary Tables S1 to S5 from the bundled source CSV files:

```bash
python -m exposure_aligned_tte.build_simulation_supplementary_tables
```

Rebuild Figure 3 from the bundled source CSV files:

```bash
python -m exposure_aligned_tte.build_simulation_figure3
```

## Re-run the 200-Replicate Simulation

To regenerate the full final simulation output bundle from scratch:

```bash
python -m exposure_aligned_tte.simulation_manuscript \
  --outdir results/generated/simulation_manuscript_200reps \
  --reps 200 \
  --sensitivity-reps 200
```

## Included Manuscript Outputs

### Supplementary Tables

- Table S1. Simulation performance under simple discontinuation and intermittent use
- Table S2. Simulation representative risk curves and truth curves
- Table S3. Correlation between cumulative exposure and exposure-history weighted burden
- Table S4. Simulation weighting-function and accumulation-scale sensitivity
- Table S5. Simulation decomposition and weight diagnostics

### Figures

- Figure 1. Timing-sensitive exposure histories
- Figure 2. Composite timing-aware simulation framework
- Figure 3. Multi-panel simulation sensitivity figure

## Reproducibility Scope

This repository reproduces the public simulation analyses and simulation-derived manuscript outputs. It does not include protected electronic health record data or patient-level clinical analysis code.
