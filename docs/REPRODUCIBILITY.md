# Reproducibility and execution

## Simulation inputs

The optical and electro-thermal calculations use wavelength-dependent complex refractive indices for the device layers, an AM1.5G reference spectrum, and hourly spectral irradiance and meteorological records. Input definitions and acquisition sources are described in the manuscript and Supplementary Information.

Place the material n-k files in `data/` or select a material directory with `SimulationEngine(data_folder=...)` in `core/simulation_engine.py`. The AM1.5G loader reads CSV input from `data/nrel_meteosat_标准光谱适配模型版(25度)_2019_one_axis/`. Hourly records are read through `data_io/file_io.py`, with POA and solar geometry preparation provided by `data_io/poa_spectrum.py`.

## Environment

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

## Reference configuration

`config/parameter_manifest.json` lists the device geometry, bandgaps, temperature coefficients, thermal parameters, diode parameters and WBG scan range. `config/config_manager.py` defines the simulation settings.

## Checks

```bash
python scripts/reviewer_smoke_check.py
pytest -q
```

The smoke check verifies the parameter manifest and the required source modules. With Numba available, it evaluates a tandem electrical point and reports the calculated maximum power. The tests check the reference parameter values and Python source syntax. Checks operate on these source paths, allowing input datasets in `data/` and keeping virtual environments and caches outside the check scope.
