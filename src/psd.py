"""
Power Spectral Density (PSD) computation via Welch's method.

"""

import numpy as np
from scipy import signal


def _detect_fs(t) -> float:
    """Infer sampling frequency (Hz) from a time vector (datetime or numeric hours)."""
    t = np.asarray(t)
    dt = t[2] - t[1]
    if hasattr(dt, "total_seconds"):               # pd.Timedelta
        dt = dt.total_seconds()
    elif np.issubdtype(type(dt), np.timedelta64):   # np.timedelta64
        dt = dt / np.timedelta64(1, "s")
    elif dt > 0.5:                                  # numeric, likely hours
        dt = dt * 3600
    return 1.0 / dt


def compute_psd(s, fs: float, nperseg: int | None = None):
    """
    Compute Welch PSD for a single temperature signal.

    Parameters
    ----------
    s       : array-like – temperature signal
    fs      : float      – sampling frequency (Hz)
    nperseg : int | None  – Welch segment length (default: N // 2)

    Returns
    -------
    freqs, psd : 1-D ndarrays  (DC component removed)
    """
    s = np.asarray(s, dtype=float)
    N = len(s)
    if nperseg is None:
        nperseg = N // 2
    freqs, psd = signal.welch(
        s - np.mean(s), fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window="hann",
    )
    return freqs[1:], psd[1:]  # drop DC


def compute_psd_human(df_human, nperseg: int | None = None):
    """
    Compute Welch PSD for the human CBT data.

    Parameters
    ----------
    df_human : DataFrame – must contain 'temp_interp' and 'datetime' columns
    nperseg  : int | None

    Returns
    -------
    freqs, psd, fs, nperseg : ndarrays + float + int
    """
    fs = _detect_fs(df_human["datetime"])
    s = df_human["temp_interp"].values
    N = len(s)
    if nperseg is None:
        nperseg = N // 2
    freqs, psd = compute_psd(s, fs, nperseg)
    return freqs, psd, fs, nperseg


def compute_psd_all_species(species_data_clean: dict):
    """
    Compute Welch PSD for every individual in every species.

    Parameters
    ----------
    species_data_clean : dict[str, DataFrame]
        Keys are species names; each DataFrame has an 'Hour' column and
        one column per individual.

    Returns
    -------
    psd_results : dict[str, dict]
        {species_name: {
            'fs'          : float,
            'individuals' : {col_name: (freqs, psd), ...}
        }}
    """
    psd_results: dict = {}
    for species_name, df in species_data_clean.items():
        animal_cols = [c for c in df.columns if c != "Hour"]
        hours = df["Hour"].values
        dt_s = float(np.median(np.diff(hours))) * 3600  # hours → seconds
        fs = 1.0 / dt_s

        individuals: dict = {}
        for col in animal_cols:
            temp = df[col].dropna().values.astype(float)
            N = len(temp)
            nperseg = min(N // 2, max(256, N // 4))
            f, p = compute_psd(temp, fs, nperseg)
            individuals[col] = (f, p)

        psd_results[species_name] = {"fs": fs, "individuals": individuals}

    return psd_results
