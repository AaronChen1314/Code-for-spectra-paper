# Nature Energy reviewer code package

Core numerical implementation for the manuscript: optical transfer-matrix/scattering calculation, photon-current integration, temperature-dependent double-diode tandem solver, PVsyst thermal iteration, and POA quality-control interface.

## Data policy

No spectral data are included: no material n-k files, AM1.5G spectrum, or hourly NSRDB/NLR spectra. These are external inputs described in the manuscript and SI. Authorized reviewers can supply them via `SimulationEngine(data_folder=...)`. Missing inputs are reported explicitly; placeholder spectra are not used for publication results.

## Manuscript alignment

`parameter_manifest.json` records the cross-checked values: 450/1200 nm absorber thicknesses; 1.80/1.25 eV reference bandgaps at 298.15 K; +0.31/+0.72 meV K-1 bandgap coefficients; U0=14.58 W m-2 K-1; U1=4.37 W s m-3 K-1; fitted double-diode parameters; beta1=3.0 and beta2=2.5; and a 1.70-2.00 eV WBG scan in 1 meV steps.

## Reviewer smoke check

Run `python reviewer_smoke_check.py`. It checks the parameter manifest, the tandem electrical solver at an in-memory photocurrent point, and confirms that no spectral data files are present. The production analysis requires authorized external spectral inputs.

## Environment

Python 3.10+; NumPy, SciPy, pandas, pvlib, numba and PyYAML.
