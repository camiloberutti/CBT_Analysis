"""
Plotting functions for CBT analysis.

Includes:
  - Time-series plots
  - Fourier / PSD log-log plots
  - Species exploration with outlier flagging
  - Before / after outlier comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal


# ── Time-series ──────────────────────────────────────────────────────────────

def plot_timeserie(datetime, temp, specie: str) -> None:
    """Plot a single temperature time series."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(datetime, temp, linewidth=0.3, alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Core Body Temperature (°C)")

    if specie == "Human":
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(),
                 rotation=45, ha="right", fontweight="bold")
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, ha="right")

    ax.set_title(f"Core Body Temperature (CBT) — {specie}")
    plt.tight_layout()
    plt.show()


# ── Species exploration (all individuals) ────────────────────────────────────

def plot_species_exploration(species_data: dict[str, pd.DataFrame]) -> None:
    """
    Plot every individual per species, flagging >4σ outliers.
    """
    for species_name, df_sp in species_data.items():
        animal_cols = [c for c in df_sp.columns if c != "Hour"]
        n_animals = len(animal_cols)

        fig, axes = plt.subplots(n_animals, 1, figsize=(
            14, 3 * n_animals), sharex=True)
        if n_animals == 1:
            axes = [axes]

        fig.suptitle(
            f"{species_name} — All individuals ({n_animals})",
            fontsize=14, fontweight="bold",
        )
        hours = df_sp["Hour"].values

        for i, col in enumerate(animal_cols):
            temp = df_sp[col].values.astype(float)
            ax = axes[i]
            ax.plot(hours, temp, linewidth=0.5, alpha=0.8)
            ax.set_ylabel("T (°C)", fontsize=9)
            ax.set_title(col, fontsize=10, loc="left")
            ax.grid(True, alpha=0.3)

            valid = temp[~np.isnan(temp)]
            if len(valid) > 0:
                median_t = np.median(valid)
                std_t = np.std(valid)
                outlier_mask = np.abs(temp - median_t) > 4 * std_t
                if outlier_mask.any():
                    ax.scatter(
                        hours[outlier_mask], temp[outlier_mask],
                        color="red", s=15, zorder=5,
                        label=f"{outlier_mask.sum()} outliers (>4σ)",
                    )
                    ax.legend(fontsize=8, loc="upper right")

                ax.text(
                    0.99, 0.02,
                    f"mean={median_t:.2f}°C  std={std_t:.2f}°C  NaN={np.isnan(temp).sum()}",
                    transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="wheat", alpha=0.5),
                )

        axes[-1].set_xlabel("Time (hours)")
        plt.tight_layout()
        plt.show()
        print(f"{'─' * 80}\n")


# ── Before / After outlier comparison ────────────────────────────────────────

def plot_outlier_comparison(
    species_data: dict[str, pd.DataFrame],
    species_data_clean: dict[str, pd.DataFrame],
) -> None:
    """Plot before/after comparison for species that had outliers replaced."""
    for species_name in species_data:
        df_orig = species_data[species_name]
        df_clean = species_data_clean[species_name]
        animal_cols = [c for c in df_orig.columns if c != "Hour"]
        hours = df_orig["Hour"].values

        changed_cols = [
            col for col in animal_cols
            if not np.allclose(
                df_orig[col].values.astype(float),
                df_clean[col].values.astype(float),
                equal_nan=True,
            )
        ]

        if not changed_cols:
            continue

        fig, axes = plt.subplots(
            len(changed_cols), 1,
            figsize=(14, 3.5 * len(changed_cols)), sharex=True,
        )
        if len(changed_cols) == 1:
            axes = [axes]

        fig.suptitle(
            f"{species_name} — Before / After outlier interpolation",
            fontsize=14, fontweight="bold",
        )

        for i, col in enumerate(changed_cols):
            ax = axes[i]
            orig = df_orig[col].values.astype(float)
            clean = df_clean[col].values.astype(float)

            diff_mask = orig != clean
            diff_mask = diff_mask & ~(np.isnan(orig) & np.isnan(clean))

            ax.plot(hours, orig, linewidth=0.4, alpha=0.5,
                    color="red", label="Original")
            ax.plot(hours, clean, linewidth=0.5, alpha=0.9,
                    color="blue", label="Cleaned")
            if diff_mask.any():
                ax.scatter(
                    hours[diff_mask], clean[diff_mask],
                    color="green", s=20, zorder=5, marker="x",
                    label=f"{diff_mask.sum()} interpolated pts",
                )
            ax.set_ylabel("T (°C)", fontsize=9)
            ax.set_title(col, fontsize=10, loc="left")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (hours)")
        plt.tight_layout()
        plt.show()


# ── Fourier / PSD helpers ────────────────────────────────────────────────────

def plot_spectral_log(s, t, NFFT: int, specie: str, log: bool = True) -> None:
    """Plot time series + PSD (Welch) side by side."""
    dt = t.iloc[2] - t.iloc[1]
    if isinstance(dt, pd.Timedelta):
        dt = dt.total_seconds()
    Fs = 1 / dt

    fig, (ax0, ax1) = plt.subplots(2, 1, layout="constrained")
    ax0.plot(t.values, s.values)
    ax0.set_xlabel("Time")
    ax0.set_ylabel("T [ºC]")
    ax0.set_title(f"SP {specie}")

    freqs, psd = signal.welch(s.values, fs=Fs, nperseg=NFFT)

    if log:
        ax1.loglog(freqs[1:], psd[1:])
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("PSD")
        ax1.set_title("Power Spectral Density (log-log)")
    else:
        ax1.plot(freqs, psd)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("PSD")
        ax1.set_title("Power Spectral Density")

    ax1.grid(True, alpha=0.3)
    plt.show()


# ── PSD log-log with circadian markers ───────────────────────────────────────

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


def plot_psd_single(freqs, psd, fs, specie="Signal", nperseg=None, ax=None):
    """Plot a single PSD curve on a log-log axis with circadian markers."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 5))

    ax.loglog(freqs, psd, linewidth=0.8, color="steelblue")
    _add_circadian_markers(ax, freqs)

    title = f"PSD (log-log) — {specie}  [fs = {fs:.4g} Hz"
    if nperseg is not None:
        title += f", nperseg = {nperseg}"
    title += "]"

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("PSD (°C² / Hz)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    if own_fig:
        plt.tight_layout()
        plt.show()
    return ax


def plot_psd_species_overlay(species_psd_results: dict, species_name: str) -> None:
    """
    Overlay all individuals of one species on a single log-log PSD plot.

    Parameters
    ----------
    species_psd_results : dict
        {"fs": float, "individuals": {col: (freqs, psd)}}
    species_name : str
    """
    info = species_psd_results
    fs = info["fs"]
    indiv = info["individuals"]
    n = len(indiv)
    cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (col, (freqs, psd)) in enumerate(indiv.items()):
        colour = cmap(i / max(n - 1, 1))
        ax.loglog(freqs, psd, linewidth=0.7,
                  alpha=0.6, color=colour, label=col)

    first_freqs = next(iter(indiv.values()))[0]
    _add_circadian_markers(ax, first_freqs)

    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("PSD (°C² / Hz)", fontsize=12)
    ax.set_title(
        f"PSD (log-log) — {species_name}  "
        f"[{n} individuals, fs = {fs:.4g} Hz]",
        fontsize=13,
    )
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
