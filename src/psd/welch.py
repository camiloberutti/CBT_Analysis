"""Welch PSD: scipy.signal.welch wrapper for CBT signals."""

import numpy as np
from scipy import signal


def compute_welch_psd(
    s,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
):
    """
    Compute Welch PSD for a single temperature signal.

    Parameters
    ----------
    s        : 1D array-like — input signal.
    fs       : float — sampling frequency (Hz).
    nperseg  : int or None — segment length (defaults to len(s)).
    noverlap : int or None — overlap (defaults to nperseg // 2).
    window   : str — window function name (default "hann").

    Returns
    -------
    freqs, psd : 1D arrays (DC component removed).
    """
    s = np.asarray(s, dtype=float)
    N = len(s)
    if nperseg is None:
        nperseg = N
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, psd = signal.welch(
        s - np.mean(s), fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
    )
    return freqs[1:], psd[1:]  # drop DC
