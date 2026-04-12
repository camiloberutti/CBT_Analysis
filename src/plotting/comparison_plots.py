"""Comparison and IRASA separation summary plots."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

from src.plotting.psd_plots import _add_circadian_markers


def plot_psd_comparison(specie_name: str, method_results: dict, fs: float):
    """
    Plots PSD + piecewise fit for all three methods in a single figure (1 row, 3 cols).
    """
    methods = ["maczak", "welch", "irasa"]
    colors = {"maczak": "steelblue",
              "welch": "darkorange", "irasa": "seagreen"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle(f"{specie_name} — PSD method comparison",
                 fontsize=14, fontweight="bold")

    for ax, method in zip(axes, methods):
        res = method_results.get(method)
        ax.set_title(method.upper())
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")

        if res is None:
            ax.text(0.5, 0.5, "Failed", ha="center",
                    va="center", transform=ax.transAxes)
            continue

        freqs, PSD, my_pwlf = res["freqs"], res["PSD"], res["pwlf"]

        # raw PSD
        ax.loglog(freqs, PSD, color=colors[method],
                  alpha=0.6, linewidth=1, label="PSD")

        # piecewise fit
        x_fit_log = np.logspace(np.log10(freqs[0]), np.log10(freqs[-1]), 500)
        y_fit = np.exp(my_pwlf.predict(np.log(x_fit_log)))
        ax.loglog(x_fit_log, y_fit, "k--", linewidth=1.8, label="PWLF fit")

        # breakpoints (in log space)
        for bp_log in my_pwlf.fit_breaks[1:-1]:
            ax.axvline(np.exp(bp_log), color="red", linestyle=":",
                       linewidth=1.2, label="breakpoint")

        # annotate slopes
        for i, slope in enumerate(my_pwlf.slopes):
            ax.annotate(
                f"β={slope:.2f}",
                xy=(0.05 + i * 0.35, 0.08),
                xycoords="axes fraction",
                fontsize=9,
                color="black",
            )

        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def PSD_separation_summary_plot(irasa_freqs, psd_aperiodic, psd_periodic, pwlf_obj, oscillatory_log, oscillatory_smoothed, prominence=0.1):
    """
    Generates a 6-panel summary figure showing the IRASA separation, 
    piecewise fractal fit, and oscillatory component extraction.
    """
    y_total = np.log(psd_aperiodic + psd_periodic)
    y_fractal_fit = pwlf_obj.predict(np.log(irasa_freqs))

    # 3. Create the multi-panel plot
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Top Left: Total PSD
    axes[0, 0].loglog(irasa_freqs, np.exp(y_total), color='black')
    axes[0, 0].set_title("Total PSD")
    axes[0, 0].set_ylabel("Power")
    axes[0, 0].grid(True, which="both", alpha=0.3)
    _add_circadian_markers(axes[0, 0], irasa_freqs)

    # Top Right: Legend Panel
    axes[0, 1].axis('off')
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Total PSD'),
        Line2D([0], [0], color='#1f4e79', lw=2, label='Fractal PSD (IRASA)'),
        Line2D([0], [0], color='#1f4e79', lw=2,
               ls='-', label='Piecewise Linear Fit'),
        Line2D([0], [0], color='gray', lw=1,
               ls='--', label='PWLF Breakpoints'),
        Line2D([0], [0], color='#6b3e23', lw=1, label='Oscillatory (Raw)'),
        Line2D([0], [0], color='#6b3e23', lw=2,
               label='Oscillatory (Smoothed)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=8, label='Detected Peaks'),
        Line2D([0], [0], color='blue', lw=1, alpha=0.5,
               label='Circadian Markers (24h, 12h, ...)'),
    ]
    axes[0, 1].legend(handles=legend_elements, loc='center',
                      title="Figure Legend", fontsize=10, frameon=True)

    # Middle Left: Fractal PSD (IRASA)
    axes[1, 0].loglog(irasa_freqs, psd_aperiodic, color='#1f4e79')
    axes[1, 0].set_title("Fractal PSD (IRASA)")
    axes[1, 0].set_ylabel("Power")
    axes[1, 0].grid(True, which="both", alpha=0.3)
    _add_circadian_markers(axes[1, 0], irasa_freqs)

    # Middle Right: Piecewise Fit
    axes[1, 1].loglog(irasa_freqs, psd_aperiodic, color='#1f4e79', alpha=0.3)
    axes[1, 1].loglog(irasa_freqs, np.exp(y_fractal_fit),
                      color='#1f4e79', linewidth=2)
    internal_breaks = pwlf_obj.fit_breaks[1:-1]
    for b in internal_breaks:
        axes[1, 1].axvline(np.exp(b), color='gray', linestyle='--')
    axes[1, 1].set_title("Piecewise Linear Fit")
    axes[1, 1].set_ylabel("Power")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    _add_circadian_markers(axes[1, 1], irasa_freqs)

    # Bottom Left: Oscillatory PSD (Residuals)
    axes[2, 0].loglog(irasa_freqs, np.exp(oscillatory_log), color='#6b3e23')
    axes[2, 0].set_title("Oscillatory PSD (Residuals)")
    axes[2, 0].set_xlabel("f [Hz]")
    axes[2, 0].set_ylabel("Power")
    axes[2, 0].grid(True, which="both", alpha=0.3)
    _add_circadian_markers(axes[2, 0], irasa_freqs)

    # Bottom Right: Smoothed Oscillatory + Peak Detection
    axes[2, 1].loglog(irasa_freqs, np.exp(oscillatory_log),
                      color='#6b3e23', alpha=0.2)
    axes[2, 1].loglog(irasa_freqs, np.exp(
        oscillatory_smoothed), color='#6b3e23', linewidth=2)

    # Peaks markers
    peaks, _ = find_peaks(oscillatory_smoothed, prominence=prominence)
    axes[2, 1].plot(irasa_freqs[peaks], np.exp(
        oscillatory_smoothed[peaks]), "ro")

    _add_circadian_markers(axes[2, 1], irasa_freqs)
    axes[2, 1].set_title("Gaussian Smoothing & Peaks")
    axes[2, 1].set_xlabel("f [Hz]")
    axes[2, 1].set_ylabel("Power")
    axes[2, 1].grid(True, which="both", alpha=0.3)

    plt.show()
