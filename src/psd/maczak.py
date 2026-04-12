"""Maczak-style PSD: DFT-based single-sided PSD with logarithmic binning.

Reference: Maczak et al. normalization.
"""

import numpy as np


def dft_single_sided_psd(x, fs: float):
    """
    Calculates single sided PSD based on Maczak et al.
     - x: 1D array of signal values (e.g., temperature)
     - fs: sampling frequency in Hz (e.g., 1/3600 for hourly data)
    Returns:
     - freqs: array of frequency bins (Hz)
     - psd: array of PSD values (°C²/Hz)"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    # Subtract the mean to avoid a giant peak at frequency 0 (DC)
    x = x - np.mean(x)

    N = x.size
    dt = 1.0 / float(fs)
    T = N * dt

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=dt)

    # power calculation: |X|^2 * T / N^2
    psd = (np.abs(X) ** 2) * T / (N**2)

    # factor 2 for frequencies > 0 (except Nyquist if N even)
    if psd.size > 1:
        psd[1:] *= 2.0

    return freqs, psd


def _log_bin_edges(fmin: float, fmax: float, bins_per_decade: int):
    """Creates the edges of the bins in logarithmic scale."""
    decades = np.log10(fmax) - np.log10(fmin)
    n_bins = max(1, int(np.ceil(decades * bins_per_decade)))
    return np.logspace(np.log10(fmin), np.log10(fmax), n_bins + 1)


def _bin_with_edges(freqs, values, edges, mode: str = "average"):
    """Group PSD values into bins defined by edges, using average or sum."""
    mask = (freqs > 0) & np.isfinite(values)
    f_clean = freqs[mask]
    v_clean = values[mask]

    out_f, out_v = [], []
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        # Bin range definition
        in_bin = (f_clean >= left) & (f_clean < right) if i < len(
            edges)-2 else (f_clean >= left) & (f_clean <= right)

        if not np.any(in_bin):
            continue

        # Bin central frequency (geometric mean)
        f_geo = np.sqrt(left * right)

        # Bin value (average in LBBA, addition for LBBS)
        val = np.mean(v_clean[in_bin]) if mode == "average" else np.sum(
            v_clean[in_bin])

        out_f.append(f_geo)
        out_v.append(val)

    return np.asarray(out_f), np.asarray(out_v)


def compute_maczak_psd(signals, fs, bins_per_decade=10):
    """
    Main workflow to obtain the ensemble PSD (average of subjects). Or smoother PSD if only one subject.
     - signals: list of 1D arrays, each containing the signal for one subject
     - fs: sampling frequency in Hz
     - bins_per_decade: how many bins to use per decade in the log-binning
    Returns:
     - freqs: array of frequency bins (Hz)
     - psd: array of ensemble-averaged PSD values (°C²/Hz)"""

    subject_data = []
    for s in signals:
        f, S = dft_single_sided_psd(s, fs)
        # Remove 0 freq
        subject_data.append((f[1:], S[1:]))

    # Common freq range for all subjects
    fmin = max(np.min(f) for f, _ in subject_data)
    fmax = min(np.max(f) for f, _ in subject_data)
    edges = _log_bin_edges(fmin, fmax, bins_per_decade)

    binned_psds = []
    common_f = None

    for f_raw, S_raw in subject_data:
        # Log Binning
        f_bin, S_bin = _bin_with_edges(f_raw, S_raw, edges, mode="average")
        # Normalization (Integral = 1)
        ssum = np.sum(S_bin)
        S_norm = S_bin / ssum if ssum > 0 else S_bin

        binned_psds.append(S_norm)
        if common_f is None:
            common_f = f_bin

    # ensamble average
    S_ensemble = np.mean(np.vstack(binned_psds), axis=0)

    return common_f, S_ensemble
