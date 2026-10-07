"""Validation and in-memory repair for NREL one-axis POA spectra."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pvlib

try:
    from scipy.integrate import trapezoid
except Exception:
    trapezoid = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

class PureNumpyKNN:
    """Pure NumPy fallback for k-nearest neighbor query."""
    def __init__(self, data):
        self.data = np.asarray(data)
    def query(self, x, k=1):
        x = np.asarray(x)
        if x.ndim == 1:
            dists = np.linalg.norm(self.data - x, axis=1)
            idx = np.argsort(dists)[:k]
            return dists[idx], idx
        dists = np.linalg.norm(self.data[None, :, :] - x[:, None, :], axis=-1)
        idx = np.argsort(dists, axis=1)[:, :k]
        sorted_dists = np.take_along_axis(dists, idx, axis=1)
        return sorted_dists, idx

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = PureNumpyKNN


LOGGER = logging.getLogger("simulation_engine")
TRACKER_GCR = 2.0 / 7.0


class SpectralDataError(RuntimeError):
    """Raised when an irradiated source row cannot be used or repaired safely."""


@dataclass(frozen=True)
class SpectrumAudit:
    city: str
    source_plane: str
    source_geometry_corrupt: bool
    source_azimuth_mae_deg: float
    missing_hours_before: int
    missing_hours_after: int
    missing_ghi_kwh_m2: float
    missing_ghi_fraction_pct: float
    reconstructed_hours: int
    validation_median_sam_rad: float
    validation_p95_sam_rad: float


def read_nrel_metadata(path: str) -> dict[str, str]:
    """Read the two metadata rows at the beginning of an NSRDB CSV."""
    with open(path, "r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        keys = next(reader)
        values = next(reader)
    return dict(zip(keys, values))


def _column(df: pd.DataFrame, col_map: dict, key: str, default: float) -> np.ndarray:
    name = col_map.get(key)
    if name is None:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy(float)


def _utc_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    required = ["Year", "Month", "Day", "Hour", "Minute"]
    if not all(name in df.columns for name in required):
        raise SpectralDataError(f"Missing timestamp columns: {required}")
    times = pd.to_datetime(df[required].rename(columns=str.lower), errors="raise", utc=True)
    index = pd.DatetimeIndex(times)
    if len(index) not in (8760, 8784) or index.duplicated().any():
        raise SpectralDataError(
            f"Expected 8760 or 8784 unique UTC hours, found rows={len(index)}, "
            f"duplicates={int(index.duplicated().sum())}")
    year = int(df["Year"].iloc[0])
    expected = pd.date_range(f"{year}-01-01", periods=len(index), freq="h", tz="UTC")
    if not index.equals(expected):
        raise SpectralDataError(f"Timestamp sequence is not the complete {year} UTC hourly index")
    return index


def reference_tracker_geometry(df: pd.DataFrame, col_map: dict, metadata: dict):
    """Recompute solar position and the NREL one-axis geometry independently."""
    times = _utc_index(df)
    latitude = float(metadata["Latitude"])
    longitude = float(metadata["Longitude"])
    temperature = _column(df, col_map, "T_air", 12.0)
    pressure_mbar = pd.to_numeric(df.get("Pressure", 1013.25), errors="coerce")
    if np.isscalar(pressure_mbar):
        pressure_pa = np.full(len(df), float(pressure_mbar) * 100.0)
    else:
        pressure_pa = pressure_mbar.fillna(1013.25).to_numpy(float) * 100.0
    solar = pvlib.solarposition.get_solarposition(
        times, latitude, longitude, pressure=pressure_pa, temperature=temperature)
    tracker = pvlib.tracking.singleaxis(
        solar["apparent_zenith"], solar["azimuth"], axis_tilt=0.0,
        axis_azimuth=180.0, max_angle=90.0, backtrack=True, gcr=TRACKER_GCR)
    return times, solar, tracker


def _reference_poa(df, col_map, times, solar, tracker):
    ghi = _column(df, col_map, "GHI", 0.0)
    dni = _column(df, col_map, "DNI", 0.0)
    dhi = _column(df, col_map, "DHI", 0.0)
    albedo = _column(df, col_map, "Albedo", 0.2)
    zenith = solar["apparent_zenith"].to_numpy(float)
    azimuth = solar["azimuth"].to_numpy(float)
    tilt = tracker["surface_tilt"].fillna(0.0).to_numpy(float)
    panel_azimuth = tracker["surface_azimuth"].fillna(180.0).to_numpy(float)
    dni_extra = pvlib.irradiance.get_extra_radiation(times).to_numpy(float)
    airmass = pvlib.atmosphere.get_relative_airmass(zenith)
    poa = pvlib.irradiance.get_total_irradiance(
        tilt, panel_azimuth, zenith, azimuth, dni, ghi, dhi,
        dni_extra=dni_extra, airmass=airmass, albedo=albedo, model="perez")
    return np.maximum(0.0, np.nan_to_num(np.asarray(poa["poa_global"], dtype=float))), dni_extra


def _feature_matrix(df, col_map, times, zenith):
    ghi = _column(df, col_map, "GHI", 0.0)
    dni = _column(df, col_map, "DNI", 0.0)
    dhi = _column(df, col_map, "DHI", 0.0)
    temp = _column(df, col_map, "T_air", 20.0)
    precip = pd.to_numeric(df.get("Precipitable Water", 1.5), errors="coerce")
    if np.isscalar(precip):
        precip = np.full(len(df), float(precip))
    else:
        precip = precip.fillna(float(precip.median())).to_numpy(float)
    day = times.dayofyear.to_numpy(float)
    features = np.column_stack([
        np.clip(zenith, 0.0, 90.0),
        dni / np.maximum(ghi, 5.0),
        dhi / np.maximum(ghi, 5.0),
        precip,
        temp,
        np.sin(2.0 * np.pi * day / 365.0),
        np.cos(2.0 * np.pi * day / 365.0),
    ])
    center = np.nanmedian(features, axis=0)
    scale = np.nanpercentile(features, 75, axis=0) - np.nanpercentile(features, 25, axis=0)
    scale[scale < 1e-9] = 1.0
    return np.nan_to_num((features - center) / scale)


def _validate_analog_reconstruction(features, normalized, candidate_indices):
    """Leave-one-out validation of the atmospheric-state spectral analog model."""
    sample_count = min(256, len(candidate_indices))
    sample = candidate_indices[np.linspace(0, len(candidate_indices) - 1,
                                           sample_count, dtype=int)]
    tree = cKDTree(features[candidate_indices])
    distances, positions = tree.query(features[sample], k=min(9, len(candidate_indices)))
    if positions.ndim == 1:
        positions = positions[:, None]
        distances = distances[:, None]
    predicted = []
    observed = normalized[sample]
    for row, own_index in enumerate(sample):
        neighbors = candidate_indices[np.atleast_1d(positions[row])]
        dist = np.atleast_1d(distances[row])
        keep = neighbors != own_index
        neighbors, dist = neighbors[keep][:8], dist[keep][:8]
        if len(neighbors) == 0:
            raise SpectralDataError("Insufficient independent spectra for validation")
        weight = 1.0 / np.maximum(dist, 1e-9)
        weight /= weight.sum()
        predicted.append(np.sum(normalized[neighbors] * weight[:, None], axis=0))
    predicted = np.asarray(predicted)
    dot = np.sum(predicted * observed, axis=1)
    norms = np.linalg.norm(predicted, axis=1) * np.linalg.norm(observed, axis=1)
    sam = np.arccos(np.clip(dot / np.maximum(norms, 1e-30), -1.0, 1.0))
    median, p95 = float(np.median(sam)), float(np.percentile(sam, 95))
    if median > 0.15 or p95 > 1.00:
        raise SpectralDataError(
            f"Spectral reconstruction validation failed: median SAM={median:.4f}, "
            f"p95 SAM={p95:.4f} rad")
    return median, p95


def prepare_one_axis_poa(city, source_path, df, wavelengths, spectra, col_map):
    """Validate one-axis POA data and repair source-geometry failures in memory.

    Valid NREL spectral-on-demand rows are used directly. If a source geometry
    defect creates zero spectra during broadband-irradiated hours, a deterministic
    atmospheric-state analog model reconstructs spectral shape and is accepted
    only after leave-one-out validation; broadband magnitude comes from an
    independent pvlib one-axis Perez calculation using the correct coordinates.
    """
    metadata = read_nrel_metadata(source_path)
    if metadata.get("Time Zone") != "0":
        raise SpectralDataError(f"{city}: expected UTC source timestamps")
    times, solar, tracker = reference_tracker_geometry(df, col_map, metadata)
    spectra = np.asarray(spectra, dtype=float).copy()
    integrals = trapezoid(spectra, wavelengths, axis=1)
    ghi = _column(df, col_map, "GHI", 0.0)
    poa_reference, _ = _reference_poa(df, col_map, times, solar, tracker)
    # Spectrum completeness is defined on the tracker POA plane, not by GHI.
    # A few horizon rows can have rounded positive GHI while the corrected
    # solar geometry and tracker POA are exactly zero; those are not missing
    # incident POA energy and must remain zero rather than be fabricated.
    daylight = poa_reference > 1e-6
    missing = daylight & (integrals <= 1e-9)

    source_azimuth = _column(df, col_map, "Solar_Azimuth", np.nan)
    reference_azimuth = solar["azimuth"].to_numpy(float)
    angle_error = np.abs((source_azimuth - reference_azimuth + 180.0) % 360.0 - 180.0)
    valid_angle = (poa_reference >= 5.0) & np.isfinite(source_azimuth)
    azimuth_mae = float(np.mean(angle_error[valid_angle])) if valid_angle.any() else np.inf
    geometry_corrupt = not np.isfinite(azimuth_mae) or azimuth_mae > 0.1

    reconstructed = 0
    median_sam = p95_sam = 0.0
    if geometry_corrupt or missing.any():
        candidates = np.where((integrals >= 5.0) & (poa_reference >= 5.0))[0]
        # Analog reconstruction only needs a representative set of valid
        # daylight spectra; 50 is a conservative reproducibility floor and
        # avoids rejecting high-latitude/desert files with sparse valid POA.
        if len(candidates) < 50:
            raise SpectralDataError(
                f"{city}: insufficient valid spectra for repair ({len(candidates)} < 50)")
        normalized = np.zeros_like(spectra)
        normalized[candidates] = spectra[candidates] / integrals[candidates, None]
        features = _feature_matrix(
            df, col_map, times, solar["apparent_zenith"].to_numpy(float))
        median_sam, p95_sam = _validate_analog_reconstruction(
            features, normalized, candidates)

        if geometry_corrupt:
            rescale = candidates[poa_reference[candidates] >= 0.0]
            spectra[rescale] *= (poa_reference[rescale] /
                                 np.maximum(integrals[rescale], 1e-12))[:, None]

        missing_targets = np.where(missing)[0]
        tree = cKDTree(features[candidates])
        distances, positions = tree.query(features[missing_targets], k=8)
        for row, target in enumerate(missing_targets):
            neighbors = candidates[np.atleast_1d(positions[row])]
            dist = np.atleast_1d(distances[row])
            weight = 1.0 / np.maximum(dist, 1e-9)
            weight /= weight.sum()
            shape = np.sum(normalized[neighbors] * weight[:, None], axis=0)
            shape_integral = trapezoid(shape, wavelengths)
            if not np.isfinite(shape_integral) or shape_integral <= 0.0:
                raise SpectralDataError(f"{city}: invalid reconstructed spectrum at row {target}")
            spectra[target] = shape / shape_integral * poa_reference[target]
        reconstructed = len(missing_targets)

    integrals_after = trapezoid(spectra, wavelengths, axis=1)
    missing_after = daylight & (integrals_after <= 1e-9)
    if missing_after.any():
        raise SpectralDataError(
            f"{city}: {int(missing_after.sum())} irradiated rows remain without spectrum")

    # All downstream AOI calculations use independently recomputed UTC geometry.
    replacements = {
        "Zenith": solar["apparent_zenith"].to_numpy(float),
        "Solar_Azimuth": solar["azimuth"].to_numpy(float),
        "Panel_Tilt": tracker["surface_tilt"].fillna(0.0).to_numpy(float),
        "Panel_Azimuth": tracker["surface_azimuth"].fillna(180.0).to_numpy(float),
    }
    for key, values in replacements.items():
        name = col_map.get(key)
        if name is None:
            name = f"__{key}"
            col_map[key] = name
        df[name] = values
    df.attrs["reference_poa"] = poa_reference

    missing_ghi = float(ghi[missing].sum() / 1000.0)
    annual_ghi = float(ghi.sum() / 1000.0)
    audit = SpectrumAudit(
        city=city, source_plane="NREL one-axis plane-of-array",
        source_geometry_corrupt=geometry_corrupt,
        source_azimuth_mae_deg=azimuth_mae,
        missing_hours_before=int(missing.sum()),
        missing_hours_after=int(missing_after.sum()),
        missing_ghi_kwh_m2=missing_ghi,
        missing_ghi_fraction_pct=(100.0 * missing_ghi / annual_ghi if annual_ghi else 0.0),
        reconstructed_hours=reconstructed,
        validation_median_sam_rad=median_sam,
        validation_p95_sam_rad=p95_sam,
    )
    LOGGER.info("Spectrum audit: %s", audit)
    return df, spectra, integrals_after, audit


def reference_fixed_tilt_geometry(df: pd.DataFrame, col_map: dict, metadata: dict):
    """Compute solar position and fixed tilt geometry."""
    times = _utc_index(df)
    latitude = float(metadata["Latitude"])
    longitude = float(metadata["Longitude"])
    temperature = _column(df, col_map, "T_air", 12.0)
    pressure_mbar = pd.to_numeric(df.get("Pressure", 1013.25), errors="coerce")
    if np.isscalar(pressure_mbar):
        pressure_pa = np.full(len(df), float(pressure_mbar) * 100.0)
    else:
        pressure_pa = pressure_mbar.fillna(1013.25).to_numpy(float) * 100.0
    solar = pvlib.solarposition.get_solarposition(
        times, latitude, longitude, pressure=pressure_pa, temperature=temperature)

    tilt_col = col_map.get("Panel_Tilt")
    azm_col = col_map.get("Panel_Azimuth")
    if tilt_col and tilt_col in df.columns:
        tilt_val = float(df[tilt_col].iloc[0])
    else:
        tilt_val = abs(latitude)
    if azm_col and azm_col in df.columns:
        azm_val = float(df[azm_col].iloc[0])
    else:
        azm_val = 180.0 if latitude >= 0 else 0.0

    return times, solar, tilt_val, azm_val


def prepare_fixed_tilt_poa(
    city: str,
    source_path: str,
    df: pd.DataFrame,
    wavelengths: np.ndarray,
    spectra: np.ndarray,
    col_map: dict,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, SpectrumAudit]:
    """
    Validate, audit and return fixed-tilt POA spectra and consistent geometry.
    """
    metadata = read_nrel_metadata(source_path)
    if metadata.get("Time Zone") != "0":
        raise SpectralDataError(f"{city}: expected UTC source timestamps")
    times, solar, tilt_val, azm_val = reference_fixed_tilt_geometry(df, col_map, metadata)
    spectra = np.asarray(spectra, dtype=float).copy()
    integrals = trapezoid(spectra, wavelengths, axis=1)
    ghi = _column(df, col_map, "GHI", 0.0)
    dni = _column(df, col_map, "DNI", 0.0)
    dhi = _column(df, col_map, "DHI", 0.0)
    albedo = _column(df, col_map, "Albedo", 0.2)
    zenith = solar["apparent_zenith"].to_numpy(float)
    azimuth = solar["azimuth"].to_numpy(float)
    tilt_arr = np.full(len(df), tilt_val, dtype=float)
    panel_azm_arr = np.full(len(df), azm_val, dtype=float)
    dni_extra = pvlib.irradiance.get_extra_radiation(times).to_numpy(float)
    airmass = pvlib.atmosphere.get_relative_airmass(zenith)
    poa_dict = pvlib.irradiance.get_total_irradiance(
        tilt_arr, panel_azm_arr, zenith, azimuth, dni, ghi, dhi,
        dni_extra=dni_extra, airmass=airmass, albedo=albedo, model="perez")
    poa_reference = np.maximum(0.0, np.nan_to_num(np.asarray(poa_dict["poa_global"], dtype=float)))

    daylight = poa_reference > 1e-6
    missing = daylight & (integrals <= 1e-9)

    source_azimuth = _column(df, col_map, "Solar_Azimuth", np.nan)
    reference_azimuth = solar["azimuth"].to_numpy(float)
    angle_error = np.abs((source_azimuth - reference_azimuth + 180.0) % 360.0 - 180.0)
    valid_angle = (poa_reference >= 5.0) & np.isfinite(source_azimuth)
    azimuth_mae = float(np.mean(angle_error[valid_angle])) if valid_angle.any() else np.inf
    geometry_corrupt = not np.isfinite(azimuth_mae) or azimuth_mae > 0.1

    reconstructed = 0
    median_sam = p95_sam = 0.0
    if geometry_corrupt or missing.any():
        candidates = np.where((integrals >= 5.0) & (poa_reference >= 5.0))[0]
        if len(candidates) < 50:
            raise SpectralDataError(
                f"{city}: insufficient valid spectra for repair ({len(candidates)} < 50)")
        normalized = np.zeros_like(spectra)
        normalized[candidates] = spectra[candidates] / integrals[candidates, None]
        features = _feature_matrix(
            df, col_map, times, solar["apparent_zenith"].to_numpy(float))
        median_sam, p95_sam = _validate_analog_reconstruction(
            features, normalized, candidates)

        if geometry_corrupt:
            rescale = candidates[poa_reference[candidates] >= 0.0]
            spectra[rescale] *= (poa_reference[rescale] /
                                 np.maximum(integrals[rescale], 1e-12))[:, None]

        missing_targets = np.where(missing)[0]
        tree = cKDTree(features[candidates])
        distances, positions = tree.query(features[missing_targets], k=8)
        for row, target in enumerate(missing_targets):
            neighbors = candidates[np.atleast_1d(positions[row])]
            dist = np.atleast_1d(distances[row])
            weight = 1.0 / np.maximum(dist, 1e-9)
            weight /= weight.sum()
            shape = np.sum(normalized[neighbors] * weight[:, None], axis=0)
            shape_integral = trapezoid(shape, wavelengths)
            if not np.isfinite(shape_integral) or shape_integral <= 0.0:
                raise SpectralDataError(f"{city}: invalid reconstructed spectrum at row {target}")
            spectra[target] = shape / shape_integral * poa_reference[target]
        reconstructed = len(missing_targets)

    integrals_after = trapezoid(spectra, wavelengths, axis=1)
    missing_after = daylight & (integrals_after <= 1e-9)
    if missing_after.any():
        raise SpectralDataError(
            f"{city}: {int(missing_after.sum())} irradiated rows remain without spectrum")

    replacements = {
        "Zenith": solar["apparent_zenith"].to_numpy(float),
        "Solar_Azimuth": solar["azimuth"].to_numpy(float),
        "Panel_Tilt": tilt_arr,
        "Panel_Azimuth": panel_azm_arr,
    }
    for key, values in replacements.items():
        name = col_map.get(key)
        if name is None:
            name = f"__{key}"
            col_map[key] = name
        df[name] = values
    df.attrs["reference_poa"] = poa_reference

    missing_ghi = float(ghi[missing].sum() / 1000.0)
    annual_ghi = float(ghi.sum() / 1000.0)
    audit = SpectrumAudit(
        city=city, source_plane="NREL fixed-tilt plane-of-array",
        source_geometry_corrupt=geometry_corrupt,
        source_azimuth_mae_deg=azimuth_mae,
        missing_hours_before=int(missing.sum()),
        missing_hours_after=int(missing_after.sum()),
        missing_ghi_kwh_m2=missing_ghi,
        missing_ghi_fraction_pct=(100.0 * missing_ghi / annual_ghi if annual_ghi else 0.0),
        reconstructed_hours=reconstructed,
        validation_median_sam_rad=median_sam,
        validation_p95_sam_rad=p95_sam,
    )
    LOGGER.info("Spectrum audit: %s", audit)
    return df, spectra, integrals_after, audit
