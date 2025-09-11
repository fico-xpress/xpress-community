# Vessel Schedule Recovery Problem — FICO Xpress Python API Implementation

## Description

This repository contains a Jupyter notebook with a Python implementation of the **Vessel Schedule Recovery Problem (VSRP)**, a real-world optimization challenge in maritime logistics. The model is formulated as a **Mixed-Integer Programming (MIP)** problem and solved using the **FICO Xpress Python API**.

The VSRP addresses disruptions in container shipping schedules and evaluates recovery strategies such as:

- Speed adjustments
- Port omission
- Port swapping
- Handling rate changes

The model uses a **time-space network** to represent vessel movements and container flows, enabling detailed scenario analysis and cost-service trade-offs.

---

## Model Variants

The code includes multiple model classes, each representing a variation of the recovery logic:

- `ImprovedVSRPModel`: model with improved delay logic
- `CorrectedVSRPModel`: model with port-specific cost penalties
- `EnhancedVSRPModel`: model with expanded speed options

Each class builds the network, defines decision variables, and applies constraints to optimize recovery strategies under different assumptions.

---

## Features

- Multi-vessel support
- Container-level delay tracking
- Strategy classification and cost breakdown
- Sensitivity analysis on trade-off parameter α
- Scenario modeling (e.g., port closure, congestion, delays)

---

## Data & Simulation

The model uses synthetic data for:

- Port sequences and distances
- Container group generation
- Vessel schedules

These can be replaced with real-world data from shipping companies.

---

## Requirements

- Python 3.9+
- Packages: `xpress`, `matplotlib`, `pandas`, `numpy`

---

## Legal

See source code files for copyright notices.

## License

The source code files in this repository are licensed under the Apache License, Version 2.0. See [LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0) for the full license text.
