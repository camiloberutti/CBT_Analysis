"""Cross-species analysis pipelines: PSD → fit → plot for every species."""

import numpy as np
import pandas as pd
from typing import Literal

from src.psd.dispatch import compute_psd_by_method
from src.parametrization.pwlf_fit import compute_pwlf_log
from src.plotting.psd_plots import plot_psd_single
from src.plotting.comparison_plots import plot_psd_comparison


def run_pwlf_pipeline(
    species_data: dict,
    species_data_clean: dict,
    df_species_meta,
    n_lines: int = 2,
    method: Literal["maczak", "welch", "irasa"] = "maczak",
    *,
    # ── Maczak ──
    bins_per_decade: int = 100,
    # ── Welch ──
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    # ── IRASA ──
    f_range: tuple | None = None,
    hset=None,
    irasa_thresh=None,
) -> dict:
    """
    Full pipeline: PSD → piecewise linear fit → plot, for every species.

    Parameters
    ----------
    method          : "maczak" | "welch" | "irasa"
    bins_per_decade : only used when method="maczak"
    nperseg         : segment length for welch (defaults to len(signal))
    noverlap        : overlap for welch
    window          : window function for welch
    f_range         : [f_min, f_max] for irasa
    hset            : resampling factors for irasa
    irasa_thresh    : periodic threshold for irasa
    """
    pwlf_species = {}

    for specie_name in species_data.keys():
        # ── data prep ────────────────────────────────────────────────────────
        df_sp = species_data_clean[specie_name]
        signal_cols = [c for c in df_sp.columns if c != "Hour"]
        signals = [
            pd.to_numeric(df_sp[c], errors="coerce").dropna().to_numpy()
            for c in signal_cols
        ]
        signals = [s for s in signals if s.size > 3]

        dt_min = df_species_meta.loc[
            df_species_meta["sheet_name"] == specie_name, "original_interval_min"
        ].iloc[0]
        fs_specie = float(1.0 / (dt_min * 60.0))

        # ── PSD ──────────────────────────────────────────────────────────────
        psd_result = compute_psd_by_method(
            signals, fs_specie,
            method=method,
            bins_per_decade=bins_per_decade,
            nperseg=nperseg,
            noverlap=noverlap,
            window=window,
            f_range=f_range,
            hset=hset,
            irasa_thresh=irasa_thresh,
        )
        freqs, PSD = psd_result["freqs"], psd_result["PSD"]

        # ── piecewise fit ─────────────────────────────────────────────────────
        my_pwlf = compute_pwlf_log(freqs, PSD, n_lines)
        pwlf_species[specie_name] = my_pwlf

        # ── plot ──────────────────────────────────────────────────────────────
        plot_psd_single(
            freqs, PSD,
            fs=fs_specie,
            specie=f"{method.capitalize()} {specie_name} - Piecewise Fit",
            pwlf_obj=my_pwlf,
        )

    return pwlf_species


def compare_psd_methods_per_species(
    species_data: dict,
    species_data_clean: dict,
    df_species_meta,
    n_lines: int = 2,
    *,
    # ── Maczak ──
    bins_per_decade: int = 100,
    # ── Welch ──
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    # ── IRASA ──
    f_range: tuple | None = None,
    hset=None,
    irasa_thresh=None,
) -> dict:
    """
    For each species, computes PSD with all three methods and plots them together.

    Returns
    -------
    results : dict
        results[specie_name][method] = {"freqs", "PSD", "pwlf", ...}
    """
    methods = ["maczak", "welch", "irasa"]
    results = {}

    for specie_name in species_data.keys():
        # ── data prep ────────────────────────────────────────────────────────
        df_sp = species_data_clean[specie_name]
        signal_cols = [c for c in df_sp.columns if c != "Hour"]
        signals = [
            pd.to_numeric(df_sp[c], errors="coerce").dropna().to_numpy()
            for c in signal_cols
        ]
        signals = [s for s in signals if s.size > 3]

        dt_min = df_species_meta.loc[
            df_species_meta["sheet_name"] == specie_name, "original_interval_min"
        ].iloc[0]
        fs_specie = float(1.0 / (dt_min * 60.0))

        results[specie_name] = {}

        # ── compute PSD + pwlf for each method ───────────────────────────────
        for method in methods:
            try:
                psd_result = compute_psd_by_method(
                    signals, fs_specie,
                    method=method,
                    bins_per_decade=bins_per_decade,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    window=window,
                    f_range=f_range,
                    hset=hset,
                    irasa_thresh=irasa_thresh,
                )
                freqs, PSD = psd_result["freqs"], psd_result["PSD"]
                my_pwlf = compute_pwlf_log(freqs, PSD, n_lines)

                results[specie_name][method] = {
                    **psd_result,
                    "pwlf": my_pwlf,
                    "fs": fs_specie,
                }
            except Exception as e:
                print(f"[{specie_name}] method '{method}' failed: {e}")
                results[specie_name][method] = None

        # ── comparison plot ───────────────────────────────────────────────────
        plot_psd_comparison(specie_name, results[specie_name], fs=fs_specie)

    return results
