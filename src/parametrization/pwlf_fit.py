"""Piecewise linear fit in log-log space using the pwlf library."""

import numpy as np
import pwlf


def compute_pwlf_log(freqs, psd, n_lines=3):
    """
    Fit piecewise linear segments in log-log space.

    Parameters
    ----------
    freqs   : 1D array — frequency vector.
    psd     : 1D array — power spectral density.
    n_lines : int — number of line segments to fit.

    Returns
    -------
    my_pwlf : pwlf.PiecewiseLinFit object with fitted parameters.
    """
    x = np.log(freqs)
    y = np.log(psd)
    my_pwlf = pwlf.PiecewiseLinFit(x, y)

    # fit the data for n line segments
    res = my_pwlf.fit(n_lines)

    return my_pwlf
