"""PSD log-log plots with circadian markers, piecewise fits, and local beta."""

import numpy as np
import matplotlib.pyplot as plt


# ── Helpers ──────────────────────────────────────────────────────────────────

def _add_circadian_markers(ax, freqs) -> None:
    """Add vertical lines at circadian harmonics (24 h, 12 h, 8 h, 4 h)."""
    markers = {
        "24 h": 1 / (24 * 3600),
        "12 h": 1 / (12 * 3600),
        "8 h":  1 / (8 * 3600),
        "4 h":  1 / (4 * 3600),
    }
    colours = ["red", "orange", "green", "purple"]
    for (lbl, f_mark), c in zip(markers.items(), colours):
        if f_mark >= freqs[0]:
            ax.axvline(f_mark, color=c, ls="--", alpha=0.7, lw=1, label=lbl)
        else:
            ax.axvline(f_mark, color=c, ls=":", alpha=0.3, lw=1,
                       label=f"{lbl} (below res.)")


# ── Single PSD ───────────────────────────────────────────────────────────────

def plot_psd_single(freqs, psd, fs, specie="Signal", nperseg=None, n_individuals=None,
                    ax=None, pwlf_obj=None, show_markers=True,
                    aperiodic=None, periodic=None):
    """Plot a single PSD curve on a log-log axis with circadian markers.

    Parameters
    ----------
    freqs        : 1D array — frequency values.
    psd          : 1D array — PSD values.
    fs           : float — sampling frequency (Hz).
    specie       : str — label for the curve.
    nperseg      : int, optional — shown in title.
    n_individuals: int, optional — shown in title.
    ax           : matplotlib Axes, optional — if None a new figure is created.
    pwlf_obj     : pwlf object, optional — piecewise linear fit overlay.
    show_markers : bool, optional — draw vertical circadian markers.
    aperiodic    : array-like, optional — aperiodic component from IRASA.
    periodic     : array-like, optional — periodic component from IRASA.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 5))

    ax.loglog(freqs, psd, linewidth=0.8, label=specie, alpha=0.7)

    if aperiodic is not None:
        ax.loglog(freqs, aperiodic, linestyle="--",
                  color="gray", label="Aperiodic", alpha=0.8)
    if periodic is not None:
        ax.loglog(freqs, periodic, linestyle="-",
                  color="orange", label="Periodic", alpha=0.8)

    if show_markers:
        _add_circadian_markers(ax, freqs)

    # --- Piecewise linear fit overlay ---
    if pwlf_obj is not None:
        mask_f = freqs > 0
        log_freq = np.log(freqs[mask_f])

        # Smooth prediction in log-space, then back-transform for loglog plot
        x_hat = np.linspace(log_freq.min(), log_freq.max(), num=1000)
        y_hat = pwlf_obj.predict(x_hat)
        ax.plot(np.exp(x_hat), np.exp(y_hat),
                color="red", linewidth=2, label="Piecewise fit", zorder=3)

        # Annotate slopes on segments
        breaks = pwlf_obj.fit_breaks
        for i in range(len(pwlf_obj.slopes)):
            # Midpoint of the segment in log-space
            mid_log_x = (breaks[i] + breaks[i+1]) / 2
            mid_log_y = pwlf_obj.predict(mid_log_x)[0]

            # Convert back to data coordinates for annotation
            ax.text(np.exp(mid_log_x), np.exp(mid_log_y), f"β={pwlf_obj.slopes[i]:.2f}",
                    color="red", fontweight='bold', fontsize=9,
                    ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

        # Breakpoints (skip first and last — data bounds)
        internal_breaks = breaks[1:-1]
        for i, b in enumerate(internal_breaks):
            ax.axvline(np.exp(b), color="blue", linestyle="--", alpha=0.6,
                       linewidth=1.2,
                       label="Breakpoint" if i == 0 else "")

    # Note: If you are plotting multiple species, the title will update to the LAST
    # species plotted in the loop. You might want to change this to a generic title
    # if you are plotting them all together.
    title = f"{specie}  [fs = {fs:.4g} Hz"
    if n_individuals is not None:
        title += f", n = {int(n_individuals)}"
    if nperseg is not None:
        title += f", nperseg = {nperseg}"
    title += "]"

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("PSD (°C² / Hz)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    mask_f = freqs > 0
    ax.set_xlim(freqs[mask_f].min(), freqs[mask_f].max())

    mask_p = psd > 0
    if np.any(mask_p):
        ymin, ymax = psd[mask_p].min(), psd[mask_p].max()
        ax.set_ylim(ymin * 0.9, ymax * 1.1)

    if own_fig:
        plt.tight_layout()
        plt.show()
    return ax


# ── Multi PSD ────────────────────────────────────────────────────────────────

def plot_psd_multi(freqs_list, psd_list, fs, species=None, nperseg=None,
                   n_individuals_list=None, ax=None, pwlf_objs=None, show_markers=True):
    """Plot multiple PSD curves on the same log-log axis.

    Parameters
    ----------
    freqs_list : list of array-like — one frequency array per species.
    psd_list   : list of array-like — one PSD array per species.
    fs         : float — sampling frequency (used in title).
    species    : list of str, optional — labels for each curve.
    nperseg    : int, optional — shown in title.
    n_individuals_list : list of int, optional — per-species counts.
    ax         : matplotlib Axes, optional.
    pwlf_objs  : list of pwlf objects, optional.
    show_markers : bool, optional.
    """
    n = len(freqs_list)
    assert len(psd_list) == n, "freqs_list and psd_list must have the same length"

    # Defaults
    if species is None:
        species = [f"Signal {i+1}" for i in range(n)]
    if pwlf_objs is None:
        pwlf_objs = [None] * n
    if n_individuals_list is None:
        n_individuals_list = [None] * n

    assert len(species) == n, "species list length must match freqs_list"
    assert len(pwlf_objs) == n, "pwlf_objs list length must match freqs_list"

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 5))

    # --- Draw circadian markers once, from the first frequency array ---
    if show_markers:
        _add_circadian_markers(ax, freqs_list[0])

    # Track global y/x limits across all species
    all_freqs_pos = []
    all_psd_pos = []

    for freqs, psd, specie, pwlf_obj, n_individuals in zip(
            freqs_list, psd_list, species, pwlf_objs, n_individuals_list):

        ax.loglog(freqs, psd, linewidth=0.8, label=specie)

        # Per-species piecewise fit overlay
        if pwlf_obj is not None:
            mask_f = freqs > 0
            log_freq = np.log(freqs[mask_f])
            x_hat = np.linspace(log_freq.min(), log_freq.max(), num=1000)
            y_hat = pwlf_obj.predict(x_hat)
            ax.plot(np.exp(x_hat), np.exp(y_hat),
                    linewidth=2, linestyle="--", label=f"Piecewise fit ({specie})", zorder=3)

            # Midpoint slope annotations
            breaks = pwlf_obj.fit_breaks
            for i in range(len(pwlf_obj.slopes)):
                mid_log_x = (breaks[i] + breaks[i+1]) / 2
                mid_log_y = pwlf_obj.predict(mid_log_x)[0]
                ax.text(np.exp(mid_log_x), np.exp(mid_log_y), f"β={pwlf_obj.slopes[i]:.2f}",
                        fontweight='bold', fontsize=8, ha='center', va='bottom',
                        bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.5))

            for i, b in enumerate(breaks[1:-1]):
                ax.axvline(np.exp(b), linestyle=":", alpha=0.6, linewidth=1.2,
                           label=f"Breakpoint ({specie})" if i == 0 else "")

        all_freqs_pos.extend(freqs[freqs > 0])
        all_psd_pos.extend(psd[psd > 0])

    # --- Unified title ---
    title = f"PSD  [fs = {fs:.4g} Hz"
    if nperseg is not None:
        title += f", nperseg = {nperseg}"
    title += "]"

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("PSD (°C² / Hz)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    # --- Unified axis limits across all species ---
    if all_freqs_pos:
        ax.set_xlim(min(all_freqs_pos), max(all_freqs_pos))
    if all_psd_pos:
        ymin, ymax = min(all_psd_pos), max(all_psd_pos)
        ax.set_ylim(ymin * 0.9, ymax * 1.1)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return ax


# ── PSD + Local Beta ─────────────────────────────────────────────────────────

def plot_psd_with_local_beta(freqs, psd, fs, specie="Signal", fit_fmin=None, fit_fmax=None, ax=None):
    """
    Plots PSD and local beta slope on a twin axis.
    """
    from src.parametrization.beta import _local_beta, _global_beta_fit

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 7))

    # --- 1. Global Beta Fit ---
    if fit_fmin is not None and fit_fmax is not None:
        fit = _global_beta_fit(
            freqs, psd, fit_fmin=fit_fmin, fit_fmax=fit_fmax)
        if not np.isnan(fit['beta']):
            fit_f = np.logspace(
                np.log10(fit['fit_fmin']), np.log10(fit['fit_fmax']), 100)
            fit_s = 10**fit['intercept'] * fit_f**(-fit['beta'])
            ax.loglog(fit_f, fit_s, 'k-.', linewidth=1.5,
                      label=f"Fit, β={fit['beta']:.2f}, R²={fit['r2']:.3f}")

            # 1/f trendline for reference
            trendline = fit_s[0] * (fit_f / fit_f[0])**(-1)
            ax.loglog(fit_f, trendline, color='violet', linestyle='-.',
                      linewidth=1.2, label='Trendline, β=1')

    # --- 2. PSD Plot ---
    ax.loglog(freqs, psd, color='steelblue', linewidth=1.5,
              label='Power Spectral Density')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude [a. u.]', color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax.grid(True, which="both", alpha=0.3)
    # Use ax.set_title(..., loc='left') if grid to avoid title overlap
    ax.set_title(f'PSD and Local Beta - {specie}', fontsize=10)

    # --- 3. Local Beta Curve (Twin Axis) ---
    f_beta, beta = _local_beta(freqs, psd)
    ax2 = ax.twinx()

    # Shaded regions for typical biological noise/activity scales
    ax2.axhspan(0.5, 1.5, color='red', alpha=0.10,
                label='Region 0.5 ≤ β ≤ 1.5')
    ax2.axhspan(0.8, 1.2, color='red', alpha=0.15,
                label='Region 0.8 ≤ β ≤ 1.2')

    ax2.semilogx(f_beta, beta, color='red', linewidth=1.2,
                 alpha=0.6, label='β exponent curve')
    ax2.set_ylabel('β [a. u.]', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-70, 100)  # Synchronize with previous user's visualization
    ax2.axhline(0, color='red', linewidth=0.5, alpha=0.3)

    # Merge legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # If in a grid, only show legend once
    if not own_fig:
        # Check if it is the first plot to avoid cluttering human grid
        if ax.get_subplotspec().is_first_col() and ax.get_subplotspec().is_first_row():
            ax.legend(lines1 + lines2, labels1 + labels2,
                      loc='lower left', fontsize=6)
    else:
        ax.legend(lines1 + labels2, labels1 + labels2,
                  loc='lower left', fontsize=8)

    if own_fig:
        plt.tight_layout()
        plt.show()
    return ax


# ── Maczak fit plot ──────────────────────────────────────────────────────────

def plot_maczak_fit(freqs, psd, fit_results, name="CBT Analysis"):
    """Visualization of a power-law fit (Global Beta)."""
    plt.figure(figsize=(8, 5))
    plt.loglog(freqs, psd, 'bo-', alpha=0.6, label='Data (Binned)')

    # Generate fit line
    f_line = np.array(fit_results['f_range'])
    s_line = 10**(fit_results['slope'] *
                  np.log10(f_line) + fit_results['intercept'])

    plt.loglog(f_line, s_line, 'r--', lw=2,
               label=f"Fit (beta={fit_results['beta']:.2f}, R²={fit_results['r_squared']:.2f})")

    plt.title(f"Power Law Fit: {name}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (a.u.)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()
