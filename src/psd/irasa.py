"""IRASA PSD: wrapper around neurodsp.aperiodic.compute_irasa.

Separates the power spectrum into aperiodic (fractal) and periodic
(oscillatory) components using Irregular Resampling Auto-Spectral Analysis.

The inner spectral method used by IRASA (welch, medfilt, wavelet, or
multitaper) is now fully configurable from the function call.
"""

import numpy as np
from typing import Literal
from neurodsp.aperiodic import compute_irasa as _neurodsp_irasa


# ── Supported spectral methods inside IRASA ────────────────────────
IRASA_SPECTRUM_METHODS = ("welch", "medfilt", "wavelet", "multitaper")

IrasaSpectrumMethod = Literal["welch", "medfilt", "wavelet", "multitaper"]


def compute_irasa_psd(
    sig,
    fs: float,
    f_range: tuple | None = None,
    hset=None,
    thresh=None,
    *,
    # ── Inner spectral-method selection ──
    spectrum_method: IrasaSpectrumMethod = "welch",
    # ── Welch-specific parameters ──
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    avg_type: str = "mean",
    # ── Medfilt-specific parameters ──
    medfilt_window: float = 0.5,
    # ── Wavelet-specific parameters ──
    wavelet_freqs=None,
    n_cycles: int = 7,
):
    """Compute IRASA separation via neurodsp.

    Parameters
    ----------
    sig              : 1D array — input time series.
    fs               : float — sampling frequency (Hz).
    f_range          : tuple (f_lo, f_hi) or None — frequency range to keep.
    hset             : 1D array or None — resampling factors (default 1.1–1.9).
    thresh           : float or None — relative threshold for periodic separation.

    spectrum_method  : str — inner method used to compute power spectra
                       during the IRASA resampling steps.
                       One of ``'welch'`` | ``'medfilt'`` | ``'wavelet'`` |
                       ``'multitaper'``.  Default ``'welch'``.

    Welch parameters (used when spectrum_method='welch'):
        nperseg      : int or None — segment length in samples.
        noverlap     : int or None — overlap in samples.
        window       : str — window function name (default ``'hann'``).
        avg_type     : str — ``'mean'`` or ``'median'`` averaging.

    Medfilt parameters (used when spectrum_method='medfilt'):
        medfilt_window : float — length of the median-filter window in
                         seconds (default 0.5).

    Wavelet parameters (used when spectrum_method='wavelet'):
        wavelet_freqs  : 1D array — frequencies at which to compute power.
        n_cycles       : int — number of cycles for Morlet wavelet
                         (default 7).

    Returns
    -------
    freqs         : 1D array
    psd_aperiodic : 1D array — fractal (1/f) component.
    psd_periodic  : 1D array — oscillatory component.

    Notes
    -----
    neurodsp defaults nperseg = int(4 * fs), which rounds to 0 for very
    low sampling rates (e.g. CBT data at ~0.003 Hz).  This wrapper
    auto-sets nperseg = len(sig) when the caller hasn't supplied one
    and the neurodsp default would be invalid.
    """
    sig = np.asarray(sig, dtype=float)

    if spectrum_method not in IRASA_SPECTRUM_METHODS:
        raise ValueError(
            f"Unknown spectrum_method '{spectrum_method}'. "
            f"Choose from {IRASA_SPECTRUM_METHODS}."
        )

    # ── Build the kwargs dict that gets forwarded to compute_spectrum ──
    spectrum_kwargs: dict = {"method": spectrum_method}

    if spectrum_method == "welch":
        # Guard against neurodsp's default nperseg = int(4*fs) → 0
        # IRASA resamples the signal down by each h factor; the most-
        # downsampled version has length ≈ len(sig) / max(hset).
        # nperseg must fit inside that shortest version, otherwise scipy
        # auto-truncates it and the spectra end up with different shapes.
        if nperseg is None:
            default_nperseg = int(4 * fs)
            if default_nperseg < 1:
                _hset = (
                    np.arange(1.1, 1.95, 0.05)
                    if hset is None
                    else np.asarray(hset)
                )
                max_h = float(np.max(_hset))
                min_resampled_len = int(len(sig) / max_h)
                nperseg = max(min_resampled_len, 1)

        if nperseg is not None:
            spectrum_kwargs["nperseg"] = nperseg
        if noverlap is not None:
            spectrum_kwargs["noverlap"] = noverlap
        spectrum_kwargs["window"] = window
        spectrum_kwargs["avg_type"] = avg_type

    elif spectrum_method == "medfilt":
        spectrum_kwargs["window"] = medfilt_window

    elif spectrum_method == "wavelet":
        if wavelet_freqs is None:
            raise ValueError(
                "wavelet_freqs must be provided when spectrum_method='wavelet'."
            )
        spectrum_kwargs["freqs"] = wavelet_freqs
        spectrum_kwargs["n_cycles"] = n_cycles

    # multitaper has no extra required parameters — just pass through.

    freqs, psd_aperiodic, psd_periodic = _neurodsp_irasa(
        sig, fs,
        f_range=f_range,
        hset=hset,
        thresh=thresh,
        **spectrum_kwargs,
    )
    return freqs, psd_aperiodic, psd_periodic
