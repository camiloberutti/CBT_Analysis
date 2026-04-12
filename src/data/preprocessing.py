"""
Preprocessing functions for CBT data.

Handles:
  - Human NaN-gap interpolation
  - Species outlier detection (>4σ) and linear-interpolation replacement
"""

import copy
import numpy as np
import pandas as pd


# ── Human preprocessing ──────────────────────────────────────────────────────

def preprocess_human(df_human: pd.DataFrame, fs: float = 1.0) -> pd.DataFrame:
    """
    Fill NaN gaps by linear interpolation.

    Adds columns to *df_human* **in place**:
      - temp_interp : interpolated temperature (°C)

    Also prints gap statistics.
    """
    temp_raw = df_human["temp_C"].values.copy()

    nan_mask = np.isnan(temp_raw)
    n_nan = nan_mask.sum()
    n_total = len(temp_raw)
    print(f"Total samples: {n_total}")
    print(f"NaN samples:   {n_nan} ({100 * n_nan / n_total:.1f}%)")

    if n_nan > 0:
        diff = np.diff(nan_mask.astype(int))
        gap_starts = np.where(diff == 1)[0] + 1
        gap_ends = np.where(diff == -1)[0] + 1
        if nan_mask[0]:
            gap_starts = np.concatenate(([0], gap_starts))
        if nan_mask[-1]:
            gap_ends = np.concatenate((gap_ends, [n_total]))
        gap_lengths = gap_ends - gap_starts
        print(f"Number of gaps: {len(gap_lengths)}")
        print(
            f"Gap lengths (samples): min={gap_lengths.min()}, "
            f"max={gap_lengths.max()}, median={np.median(gap_lengths):.0f}"
        )
        print(f"Longest gap: {gap_lengths.max() / 3600:.2f} hours")

    temp_interp = pd.Series(temp_raw).interpolate(method="linear").values
    temp_interp = pd.Series(temp_interp).bfill().ffill().values
    df_human["temp_interp"] = temp_interp

    duration_h = len(temp_interp) / fs / 3600
    print(f"\nAfter interpolation:")
    print(f"Signal length: {len(temp_interp)} samples")
    print(f"Duration: {duration_h:.1f} hours ({duration_h / 24:.1f} days)")
    print(f"Sampling rate: {fs} Hz")
    print(f"Remaining NaNs: {np.isnan(temp_interp).sum()}")
    return df_human

def split_by_longest_gap(df: pd.DataFrame, col_raw: str = "temp_C"):
    """
    Identify longest gap (NaN) and split the DataFrame into two:
    one with data before the gap and another with data after the gap.
    """
    temp_values = df[col_raw].values
    nan_mask = np.isnan(temp_values)
    
    if not nan_mask.any():
        print("No NaNs Found. Returning original DataFrame and an empty one.")
        return df, pd.DataFrame()

    # Detect gaps using the difference of the NaN mask
    diff = np.diff(nan_mask.astype(int))
    gap_starts = np.where(diff == 1)[0] + 1
    gap_ends = np.where(diff == -1)[0] + 1

    # Adjust borders if it starts or ends with NaN
    if nan_mask[0]:
        gap_starts = np.concatenate(([0], gap_starts))
    if nan_mask[-1]:
        gap_ends = np.concatenate((gap_ends, [len(temp_values)]))

    # Find the index of the longest gap
    gap_lengths = gap_ends - gap_starts
    longest_idx = np.argmax(gap_lengths)
    
    # Define the split points
    split_start = gap_starts[longest_idx]
    split_end = gap_ends[longest_idx]
    
    # Create the two segments (excluding the gap itself)
    df_before = df.iloc[:split_start].copy()
    df_after = df.iloc[split_end:].copy()
    
    print(f"Longest gap found between indices {split_start} and {split_end}")
    print(f"Gap length: {gap_lengths[longest_idx]} samples")
    print(f"Segment 1: {len(df_before)} samples | Segment 2: {len(df_after)} samples")
    
    return df_before, df_after


# ── Species outlier cleaning ─────────────────────────────────────────────────

def clean_species_outliers(
    species_data: dict[str, pd.DataFrame],
    sigma: float = 4.0,
) -> dict[str, pd.DataFrame]:
    """
    Replace outliers (> *sigma* σ from median) with linear interpolation.

    Returns
    -------
    species_data_clean : dict[str, DataFrame]
        Deep-copied DataFrames with outliers replaced.
    """
    species_data_clean: dict[str, pd.DataFrame] = {}
    total_outliers = 0

    for species_name, df_sp in species_data.items():
        df_clean = df_sp.copy()
        animal_cols = [c for c in df_clean.columns if c != "Hour"]

        for col in animal_cols:
            temp = df_clean[col].values.astype(float)
            valid = temp[~np.isnan(temp)]
            if len(valid) == 0:
                continue

            median_t = np.median(valid)
            std_t = np.std(valid)
            outlier_mask = np.abs(temp - median_t) > sigma * std_t
            n_out = outlier_mask.sum()

            if n_out > 0:
                print(
                    f"{species_name:20s} | {col:15s} | {n_out:3d} outliers replaced "
                    f"(threshold: {median_t:.2f} ± {sigma * std_t:.2f} °C)"
                )
                temp[outlier_mask] = np.nan
                df_clean[col] = (
                    pd.Series(temp).interpolate(
                        method="linear").bfill().ffill().values
                )
                total_outliers += n_out

        species_data_clean[species_name] = df_clean

    #print(f"\n{'═' * 60}")
    #print(f"Total outliers replaced across all species: {total_outliers}")
    return species_data_clean
