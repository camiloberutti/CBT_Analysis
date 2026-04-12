"""Unified PSD dispatcher — routes to Maczak, Welch, or IRASA."""

import numpy as np
from typing import Literal

from src.psd.maczak import compute_maczak_psd
from src.psd.welch import compute_welch_psd
from src.psd.irasa import compute_irasa_psd, IrasaSpectrumMethod


def compute_psd_by_method(
    signals: list,
    fs: float,
    method: Literal["maczak", "welch", "irasa"] = "maczak",
    *,
    # ── Maczak-specific ──
    bins_per_decade: int = 100,
    # ── Welch-specific ──
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    # ── IRASA-specific (neurodsp) ──
    f_range: tuple | None = None,
    hset=None,
    irasa_thresh=None,
    # ── IRASA inner spectral-method selection ──
    irasa_spectrum_method: IrasaSpectrumMethod = "welch",
    # ── IRASA-Welch parameters (when irasa_spectrum_method='welch') ──
    irasa_avg_type: str = "mean",
    # ── IRASA-Medfilt parameters (when irasa_spectrum_method='medfilt') ──
    irasa_medfilt_window: float = 0.5,
    # ── IRASA-Wavelet parameters (when irasa_spectrum_method='wavelet') ──
    irasa_wavelet_freqs=None,
    irasa_n_cycles: int = 7,
) -> dict:
    """Unified PSD dispatcher across available methods.

    Parameters
    ----------
    signals         : list of 1-D arrays.
    fs              : sampling frequency (Hz).
    method          : ``'maczak'`` | ``'welch'`` | ``'irasa'``.
    bins_per_decade : (maczak) log-binning resolution.
    nperseg         : (welch / irasa-welch) segment length.
    noverlap        : (welch / irasa-welch) overlap length.
    window          : (welch / irasa-welch) window function name.
    f_range         : (irasa) frequency range (f_lo, f_hi).
    hset            : (irasa) resampling factors.
    irasa_thresh    : (irasa) periodic-component threshold.

    irasa_spectrum_method : str — inner spectral method used *inside*
        IRASA.  One of ``'welch'`` | ``'medfilt'`` | ``'wavelet'`` |
        ``'multitaper'``.  Default ``'welch'``.
    irasa_avg_type        : str — welch averaging type (``'mean'`` or
        ``'median'``).  Only used when *irasa_spectrum_method='welch'*.
    irasa_medfilt_window  : float — median-filter window in seconds.
        Only used when *irasa_spectrum_method='medfilt'*.
    irasa_wavelet_freqs   : 1-D array — frequencies for wavelet PSD.
        Required when *irasa_spectrum_method='wavelet'*.
    irasa_n_cycles        : int — Morlet wavelet cycles.
        Only used when *irasa_spectrum_method='wavelet'*.

    Returns
    -------
    dict with keys depending on *method*:
      - ``'maczak'`` → ``{"freqs", "PSD"}``
      - ``'welch'``  → ``{"freqs", "PSD"}``
      - ``'irasa'``  → ``{"freqs", "PSD", "psd_aperiodic", "psd_periodic"}``
    """
    if method == "maczak":
        freqs, PSD = compute_maczak_psd(
            signals, fs, bins_per_decade=bins_per_decade
        )
        return {"freqs": freqs, "PSD": PSD}

    # Welch and IRASA operate on a single signal; use the first
    sig = np.asarray(signals[0])
    N = nperseg if nperseg is not None else len(sig)

    if method == "welch":
        freqs, PSD = compute_welch_psd(
            sig, fs,
            nperseg=N,
            noverlap=noverlap,
            window=window,
        )
        return {"freqs": freqs, "PSD": PSD}

    if method == "irasa":
        freqs, psd_aperiodic, psd_periodic = compute_irasa_psd(
            sig, fs,
            f_range=f_range,
            hset=hset,
            thresh=irasa_thresh,
            spectrum_method=irasa_spectrum_method,
            nperseg=N,
            noverlap=noverlap,
            window=window,
            avg_type=irasa_avg_type,
            medfilt_window=irasa_medfilt_window,
            wavelet_freqs=irasa_wavelet_freqs,
            n_cycles=irasa_n_cycles,
        )
        PSD = psd_aperiodic + psd_periodic
        return {
            "freqs": freqs,
            "PSD": PSD,
            "psd_aperiodic": psd_aperiodic,
            "psd_periodic": psd_periodic,
        }

    raise ValueError(
        f"Unknown method '{method}'. Choose 'maczak', 'welch', or 'irasa'."
    )
