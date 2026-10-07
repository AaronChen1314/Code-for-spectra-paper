# Reproducibility and reviewer access

## Production inputs

The repository contains the numerical implementation and manuscript-aligned parameters, but no optical spectra. The production engine requires: (i) wavelength-dependent complex refractive indices for the device layers, and (ii) authorized hourly spectral irradiance and meteorological records. These inputs are external to this repository and are described in the manuscript and Supplementary Information.

Use `SimulationEngine(data_folder=...)` in `core/simulation_engine.py` to point to an authorized input directory. The engine does not generate publication results when required files are absent.

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

## Checks

```bash
python scripts/reviewer_smoke_check.py
pytest -q
```

The smoke check verifies the parameter manifest and confirms that no spectral files are included. The full production analysis additionally requires authorized external inputs.
