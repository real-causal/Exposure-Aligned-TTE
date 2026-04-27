# Exposure-Aligned-TTE

Companion repository for the exposure-aligned target trial emulation framework and simulation experiments.

This repository contains the public, reproducible components of the manuscript, including simulation code, manuscript figure builders, supplementary table builders, and final public simulation outputs.

## Overview

Many real-world medication studies define treatment exposure at initiation. For long-latency outcomes, however, the relevant contrast may depend on how treatment unfolds after initiation, including persistence, discontinuation, gaps, restarts, and accumulated exposure history.

This repository supports the simulation component of an exposure-aligned target trial emulation framework designed to evaluate medication effects when post-baseline exposure history is central to the scientific question.

## Repository Structure

```text
Exposure-Aligned-TTE/
  README.md
  .gitignore
  requirements.txt
  environment.yml

  docs/
    migration_manifest.md
    reproducibility_scope.md

  src/
    exposure_aligned_tte/
      __init__.py
      README.md
      simulation_manuscript.py
      plotting.py

  scripts/
    README.md
    build_simulation_supplementary_tables.py
    build_simulation_figure_s9.py
    run_simulation_200rep.sh

  results/
    final_200rep/
      README.md
      figures/
      tables/
      source_csv/
