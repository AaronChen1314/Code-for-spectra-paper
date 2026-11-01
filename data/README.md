# Optical and irradiance inputs

The calculations use wavelength-dependent material n-k data, an AM1.5G reference spectrum, and hourly spectral irradiance and meteorological records, as described in the manuscript and Supplementary Information.

Place material n-k files here or select a material directory with `SimulationEngine(data_folder=...)`. Place the AM1.5G CSV input in `data/nrel_meteosat_标准光谱适配模型版(25度)_2019_one_axis/`. Input loading and POA preparation are provided by `data_io/file_io.py` and `data_io/poa_spectrum.py`. See `docs/REPRODUCIBILITY.md` for environment and execution instructions.
