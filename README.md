# Code for spectra paper

Core numerical code accompanying the submitted Nature Photonics manuscript. The repository is organized for editorial and peer review and contains the implementation used for optical, electrical and electro-thermal calculations.

## What is included

- `core/`: optical transfer-matrix/scattering routines and tandem double-diode/electro-thermal solvers.
- `config/`: device parameters, parameter manifest and simulation configuration.
- `data_io/`: readers and quality-control adapters for spectral and meteorological inputs.
- `utils/`: thermal, Perez sky and optical utility functions.
- `scripts/reviewer_smoke_check.py`: parameter and electrical solver checks.
- `tests/`: checks for reference parameter values and package structure.
- `docs/REPRODUCIBILITY.md`: environment, input and execution instructions.

## Device and simulation parameters

`config/parameter_manifest.json` records the reference configuration: WBG/NBG thicknesses of 450/1200 nm; reference bandgaps of 1.80/1.25 eV at 298.15 K; bandgap temperature coefficients of +0.31/+0.72 meV K-1; U0=14.58 W m-2 K-1; U1=4.37 W s m-3 K-1; fitted double-diode parameters; beta1=3.0 and beta2=2.5; and a 1.70-2.00 eV WBG scan in 1 meV steps.

## Installation and checks

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
python scripts/reviewer_smoke_check.py
pytest -q
```

The smoke check verifies the parameter manifest and package structure. With Numba available, it also evaluates one in-memory tandem electrical point. Input preparation and simulation setup are described in `docs/REPRODUCIBILITY.md`.

## Code availability

The numerical implementation and reference configuration are available in this repository for editorial and peer review.

## License and citation

See `CITATION.cff`. The associated manuscript should be cited for scientific use.
