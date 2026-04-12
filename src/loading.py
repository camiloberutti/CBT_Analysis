"""
Data loading utilities for Core Body Temperature (CBT) analysis.

Handles:
  - Human CBT data from greenTEG CORE sensor CSV
  - Multi-species data from Maloney et al. (2024) Excel workbook
"""

import numpy as np
import pandas as pd

from src.species_metadata import SPECIES_METADATA


# ── Human data ────────────────────────────────────────────────────────────────

def load_human_csv(csv_path: str, skip_rows: int = 15) -> pd.DataFrame:
    """
    Load and parse the greenTEG CORE sensor CSV.

    Returns a DataFrame with at least:
      - datetime : pd.Timestamp
      - temp_C   : float (°C), zeros replaced by NaN
    """
    df = pd.read_csv(csv_path, skiprows=skip_rows, sep=",")

    df["datetime"] = pd.to_datetime(df["time [UTC-OFS=+0200]"])
    df["temp_C"] = df["cbt [mC]"] / 1000.0  # milli-degrees → °C
    df.loc[df["temp_C"] == 0, "temp_C"] = np.nan

    return df


# ── Multi-species data ────────────────────────────────────────────────────────

def load_species_excel(xlsx_path: str) -> dict[str, pd.DataFrame]:
    """
    Load the nine-species Excel workbook.

    Returns
    -------
    species_data : dict[str, DataFrame]
        Keys are sheet / species names; each DataFrame has an 'Hour'
        column and one column per individual animal.
    """
    xl = pd.ExcelFile(xlsx_path)
    species_data = {}
    for sheet in xl.sheet_names:
        species_data[sheet] = pd.read_excel(xlsx_path, sheet_name=sheet)
    return species_data


def print_species_summary(species_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Print a formatted summary of every species (metadata + time sampling)
    and return the metadata DataFrame.
    """
    from src.species_metadata import get_species_dataframe

    print(f"Species sheets: {list(species_data.keys())}\n")

    for sheet, df_sp in species_data.items():
        meta = SPECIES_METADATA[sheet]
        bm_str = f"{meta['body_mass_kg_mean']}"
        if not np.isnan(meta["body_mass_kg_sd"]):
            bm_str += f" ± {meta['body_mass_kg_sd']}"
        print(
            f"{sheet:20s} — {meta['species_latin']:30s}  "
            f"body mass = {bm_str:>12s} kg  n={meta['n_individuals']:2d}  "
            f"Δt={meta['original_interval_min']:3g} min  "
            f"res={meta['logger_resolution_C']}°C"
        )

    df_meta = get_species_dataframe(include_human=True)
    print("\n── Full species metadata table ──")
    print(
        df_meta[
            [
                "sheet_name", "species_latin", "taxon_class",
                "body_mass_kg_mean", "body_mass_kg_sd",
                "n_individuals", "original_interval_min",
                "logger_resolution_C", "logger_accuracy_C",
            ]
        ].to_string(index=False)
    )
    return df_meta


def check_sampling(species_data: dict[str, pd.DataFrame]) -> None:
    """Print time-sampling info per species."""
    for name, df_sp in species_data.items():
        hours = df_sp["Hour"].values
        dt = np.median(np.diff(hours))
        total_h = hours[-1] - hours[0]
        print(
            f"{name:20s}  Δt = {dt*60:.1f} min  |  "
            f"total = {total_h:.0f} h  |  N = {len(hours)}"
        )
