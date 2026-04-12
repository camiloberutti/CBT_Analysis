"""Power-law (beta exponent) fitting utilities.

Includes:
  - Band selection in log-log space
  - Local beta (point-to-point slope)
  - Global beta via linear regression
  - Simple power-law fit
"""

import numpy as np
from scipy.stats import linregress


def _select_band(freqs, psd, fmin: float | None, fmax: float | None, min_points: int = 10):
    """Return log10(freq), log10(psd) restricted to a frequency band."""
    freqs = np.asarray(freqs, dtype=float)
    psd = np.asarray(psd, dtype=float)

    mask = np.isfinite(freqs) & np.isfinite(psd) & (psd > 0) & (freqs > 0)
    if fmin is not None:
        mask &= freqs >= fmin
    if fmax is not None:
        mask &= freqs <= fmax

    f_band = freqs[mask]
    p_band = psd[mask]
    if f_band.size < min_points:
        raise ValueError(
            "Not enough points in selected frequency band for power-law fit.")

    x = np.log10(f_band)
    y = np.log10(p_band)
    return x, y


def _local_beta(freqs, spectrum):
    """Compute local beta between consecutive points in log-log space."""
    freqs = np.asarray(freqs, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    valid = np.isfinite(freqs) & np.isfinite(
        spectrum) & (freqs > 0) & (spectrum > 0)
    freqs = freqs[valid]
    spectrum = spectrum[valid]
    if freqs.size < 2:
        return np.array([]), np.array([])

    logf = np.log(freqs)
    logs = np.log(spectrum)
    beta = -(np.diff(logs) / np.diff(logf))
    f_beta = np.sqrt(freqs[:-1] * freqs[1:])
    return f_beta, beta


def _global_beta_fit(freqs, spectrum, fit_fmin: float, fit_fmax: float):
    """Fit global beta from log(S) vs log(f) in a chosen frequency range."""
    freqs = np.asarray(freqs, dtype=float)
    valid_freqs = freqs[np.isfinite(freqs) & (freqs > 0)]
    if valid_freqs.size < 3:
        return {
            "beta": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
            "p_value": float("nan"),
            "n_points": int(valid_freqs.size),
            "fit_fmin": float("nan"),
            "fit_fmax": float("nan"),
        }

    # Prefer requested range, but clip to available frequencies if needed.
    fmin_eff = max(float(fit_fmin), float(valid_freqs.min()))
    fmax_eff = min(float(fit_fmax), float(valid_freqs.max()))
    if fmax_eff <= fmin_eff:
        fmin_eff = float(valid_freqs.min())
        fmax_eff = float(valid_freqs.max())

    try:
        x, y = _select_band(freqs, spectrum, fmin_eff, fmax_eff, min_points=3)
    except ValueError:
        x, y = _select_band(freqs, spectrum, None, None, min_points=3)
        fmin_eff = float(10 ** np.min(x))
        fmax_eff = float(10 ** np.max(x))

    slope, intercept, r, p, _ = linregress(x, y)
    return {
        "beta": float(-slope),
        "intercept": float(intercept),
        "r2": float(r**2),
        "p_value": float(p),
        "n_points": int(x.size),
        "fit_fmin": float(fmin_eff),
        "fit_fmax": float(fmax_eff),
    }


def fit_power_law(freqs, psd, f_range=[1e-4, 1e-2]):
    """
    Simple power-law fit (S(f) ~ f^-beta) using linear regression
    in log-log space. Useful for "Maczak-style global beta".
    """
    mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
    f_fit = freqs[mask]
    s_fit = psd[mask]

    if len(f_fit) < 2:
        return None

    # Linear regression: log(S) = -beta * log(f) + intercept
    log_f = np.log10(f_fit)
    log_s = np.log10(s_fit)

    slope, intercept, r_value, p_value, std_err = linregress(log_f, log_s)

    # The exponent beta is the negative of the slope
    beta = -slope

    return {
        "beta": beta,
        "r_squared": r_value**2,
        "f_range": f_range,
        "slope": slope,
        "intercept": intercept
    }
