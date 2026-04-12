"""Time-series plotting for CBT data."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
    Plot all individuals per species in a single plot,
    flagging >4σ outliers per individual.
    """
    for species_name, df_sp in species_data.items():
        animal_cols = [c for c in df_sp.columns if c != "Hour"]

        fig, ax = plt.subplots(figsize=(14, 6))
        fig.suptitle(
            f"{species_name} — All individuals ({len(animal_cols)})",
            fontsize=14, fontweight="bold",
        )

        hours = df_sp["Hour"].values

        for col in animal_cols:
            temp = df_sp[col].values.astype(float)

            # plot each individual
            ax.plot(hours, temp, linewidth=0.8, alpha=0.6, label=col)

            # outlier detection
            valid = temp[~np.isnan(temp)]
            if len(valid) > 0:
                median_t = np.median(valid)
                std_t = np.std(valid)
                outlier_mask = np.abs(temp - median_t) > 4 * std_t

                if outlier_mask.any():
                    ax.scatter(
                        hours[outlier_mask],
                        temp[outlier_mask],
                        color="red",
                        s=12,
                        zorder=5,
                    )

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("T (°C)")
        ax.grid(True, alpha=0.3)

        # optional: show legend only if not too many animals
        if len(animal_cols) <= 10:
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()
        print(f"{'─' * 80}\n")
