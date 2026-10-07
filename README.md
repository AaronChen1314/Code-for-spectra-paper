# Code for spectra paper

Core numerical code accompanying the submitted Nature Energy manuscript. The repository is organized for editorial and peer review and contains the implementation used for optical, electrical and electro-thermal calculations.

## What is included

- `core/`: optical transfer-matrix/scattering routines and tandem double-diode/electro-thermal solvers.
- `config/`: manuscript-aligned parameter manifest and configuration.
- `data_io/`: readers and quality-control adapters for external spectral and meteorological inputs.
- `utils/`: thermal, Perez sky and optical utility functions.
- `scripts/reviewer_smoke_check.py`: dependency-aware offline review check.
- `tests/`: lightweight checks for parameter consistency and data exclusion.
- `docs/REPRODUCIBILITY.md`: environment, input and execution instructions.

## Data availability

No spectral data are distributed here. This includes material n-k files, AM1.5G spectra and hourly NSRDB/NLR spectra. They are external inputs described in the manuscript and Supplementary Information and may be supplied to authorized reviewers through the corresponding author. The repository contains no fabricated replacement spectra.

## Manuscript alignment

`config/parameter_manifest.json` records the values cross-checked against the manuscript and SI: WBG/NBG thicknesses of 450/1200 nm; reference bandgaps of 1.80/1.25 eV at 298.15 K; bandgap temperature coefficients of +0.31/+0.72 meV K-1; U0=14.58 W m-2 K-1; U1=4.37 W s m-3 K-1; fitted double-diode parameters; beta1=3.0 and beta2=2.5; and a 1.70-2.00 eV WBG scan in 1 meV steps.

## Installation and checks

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
python scripts/reviewer_smoke_check.py
pytest -q
```

The smoke check can verify the manifest without optional numerical dependencies. With NumPy, SciPy, pandas, pvlib, numba and PyYAML installed, it also evaluates one in-memory tandem electrical point.

## Code availability

This public repository contains the review version of the code. The version associated with the manuscript is preserved in Git history. Any external input restrictions are described above and in the manuscript Code Availability statement.

## License and citation

See `CITATION.cff`. The associated manuscript should be cited for scientific use.
