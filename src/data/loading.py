"""Data loading utilities for Core Body Temperature (CBT) analysis.

Handles:
  - Human CBT data from greenTEG CORE sensor CSV files
  - Multi-species data from the workbook with one sheet per species
  - Species metadata from JSON (loaded only when needed)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


_META_DEFAULT_KEYS = [
    "sheet_name",
    "species_latin",
    "taxon_class",
    "body_mass_kg_mean",
    "body_mass_kg_sd",
    "body_mass_note",
    "n_individuals",
    "original_interval_min",
    "logger_model",
    "logger_resolution_C",
    "logger_accuracy_C",
    "measurement_site",
    "housing",
    "light_dark_cycle",
    "reference",
]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HUMAN_DATA_DIR = PROJECT_ROOT / "01_Data" / "CBT_Human"
HUMAN_ALLMETADATA = PROJECT_ROOT / "01_Data" / \
    "Paper Horvath" / "alldata.ods"
HUMAN_DATASET_DESCRIPTION = PROJECT_ROOT / "01_Data" / \
    "Paper Horvath" / "Table_Horvath_datasets.ods"

SPECIES_DATA_XLSX = PROJECT_ROOT / "01_Data" / \
    "CBT_Species" / "Five_days_of_Tc_data_in_nine_species.xlsx"
SPECIES_METADATA_JSON = PROJECT_ROOT / "01_Data" / \
    "CBT_Species" / "species_metadata.json"


def _to_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _parse_human_core_csv(csv_file: Path, skip_rows: int = 15) -> pd.DataFrame:
    """Parse one CORE CSV into a table with datetime and temp_C."""
    df = pd.read_csv(csv_file, skiprows=skip_rows)
    df["datetime"] = pd.to_datetime(
        df["time [UTC-OFS=+0200]"], errors="coerce")
    df["temp_C"] = pd.to_numeric(df["cbt [mC]"], errors="coerce") / 1000.0

    df.loc[df["temp_C"] == 0, "temp_C"] = np.nan
    df["source_file"] = csv_file.name
    df["individual_id"] = csv_file.stem
    return df


def list_human_subjects() -> list[str]:
    """Return available human subject IDs (CSV stems)."""
    return sorted(p.stem for p in HUMAN_DATA_DIR.glob("*.csv"))


def load_human_data(subject: str | None = None, skip_rows: int = 15) -> pd.DataFrame:
    """Load one human subject from 01_Data/CBT_Human.

    If subject is None, loads the first CSV file in the folder.
    """
    subjects = list_human_subjects()
    if not subjects:
        raise FileNotFoundError(
            f"No human CSV files found in {HUMAN_DATA_DIR}")

    subject_id = subjects[0] if subject is None else subject.replace(
        ".csv", "")
    csv_file = HUMAN_DATA_DIR / f"{subject_id}.csv"
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Subject file not found: {csv_file.name}. Available subjects: {subjects}"
        )

    return _parse_human_core_csv(csv_file=csv_file, skip_rows=skip_rows)


def load_human_csv(csv_path: str, skip_rows: int = 15) -> pd.DataFrame:
    """Backward-compatible wrapper to load one human CSV file."""
    return _parse_human_core_csv(csv_file=_to_path(csv_path), skip_rows=skip_rows)


def load_species_excel() -> dict[str, pd.DataFrame]:
    """Load species CBT workbook (one sheet per species)."""
    xl = pd.ExcelFile(SPECIES_DATA_XLSX)
    species_data: dict[str, pd.DataFrame] = {}
    for sheet in xl.sheet_names:
        species_data[sheet] = pd.read_excel(
            SPECIES_DATA_XLSX, sheet_name=sheet)
    return species_data


def load_species_metadata() -> pd.DataFrame:
    """Load species metadata from JSON and return a tidy DataFrame."""
    with SPECIES_METADATA_JSON.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "species" in raw and isinstance(raw["species"], dict):
        species_map = raw["species"]
    elif isinstance(raw, dict):
        species_map = raw
    else:
        raise ValueError(
            "species_metadata.json must be an object or contain a 'species' object")

    rows = []
    for sheet_name, meta in species_map.items():
        row = {"sheet_name": sheet_name}
        if isinstance(meta, dict):
            row.update(meta)
        rows.append(row)

    df_meta = pd.DataFrame(rows)
    for col in _META_DEFAULT_KEYS:
        if col not in df_meta.columns:
            df_meta[col] = np.nan

    df_meta = df_meta[_META_DEFAULT_KEYS].sort_values(
        "sheet_name").reset_index(drop=True)
    df_meta["body_mass_kg_mean"] = pd.to_numeric(
        df_meta["body_mass_kg_mean"], errors="coerce")
    positive_mass = df_meta["body_mass_kg_mean"] > 0
    df_meta["log10_body_mass_kg"] = np.nan
    df_meta.loc[positive_mass, "log10_body_mass_kg"] = np.log10(
        df_meta.loc[positive_mass, "body_mass_kg_mean"])
    return df_meta


def load_human_metadata(dataset_description: bool = False) -> pd.DataFrame:
    """Load human dataset metadata.

    Args:
        dataset_description: If True, loads HUMAN_DATASET_DESCRIPTION (Table_Horvath_datasets.ods).
                           If False, loads HUMAN_ALLMETADATA (alldata.ods).
    """
    path = HUMAN_DATASET_DESCRIPTION if dataset_description else HUMAN_ALLMETADATA
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")
    return pd.read_excel(path, engine="odf")


def get_species_meta(
    sheet_name: str,
    species_metadata_df: pd.DataFrame | None = None,
    species_df: pd.DataFrame | None = None,
) -> dict:
    """Return metadata for one species with runtime sampling fallback."""
    default_meta = {key: np.nan for key in _META_DEFAULT_KEYS}
    default_meta.update({"sheet_name": sheet_name,
                        "species_latin": sheet_name, "taxon_class": "Unknown"})

    if species_metadata_df is not None and not species_metadata_df.empty:
        match = species_metadata_df[species_metadata_df["sheet_name"] == sheet_name]
        if not match.empty:
            meta = default_meta.copy()
            meta.update(match.iloc[0].to_dict())
            return meta

    if species_df is not None:
        animal_cols = [c for c in species_df.columns if c != "Hour"]
        if animal_cols:
            default_meta["n_individuals"] = int(len(animal_cols))
        if "Hour" in species_df.columns:
            hour = pd.to_numeric(
                species_df["Hour"], errors="coerce").dropna().values
            if hour.size > 1:
                default_meta["original_interval_min"] = float(
                    np.median(np.diff(hour)) * 60.0)

    return default_meta


def print_species_summary(
    species_data: dict[str, pd.DataFrame],
    species_metadata_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Print metadata and sampling summary for each species sheet."""
    if species_metadata_df is None:
        species_metadata_df = load_species_metadata()

    # print(f"Species sheets: {list(species_data.keys())}\\n")
    # for sheet, df_sp in species_data.items():
    #     meta = get_species_meta(sheet, species_metadata_df=species_metadata_df, species_df=df_sp)
    #     bm_mean = meta.get("body_mass_kg_mean", np.nan)
    #     bm_sd = meta.get("body_mass_kg_sd", np.nan)
    #     bm_str = f"{bm_mean}" if pd.notna(bm_mean) else "nan"
    #     if pd.notna(bm_sd):
    #         bm_str += f" +/- {bm_sd}"
    #     n_individuals = meta.get("n_individuals", np.nan)
    #     n_display = int(n_individuals) if pd.notna(n_individuals) else -1
    #     print(
    #         f"{sheet:20s} - {str(meta.get('species_latin', sheet)):30s}  "
    #         f"body mass = {bm_str:>16s} kg  n={n_display:2d}  "
    #         f"dt={meta.get('original_interval_min', np.nan):3g} min  "
    #         f"res={meta.get('logger_resolution_C', np.nan)} C"
    #     )

    # print("\\n-- Full species metadata table --")
    cols = [
        "sheet_name",
        "species_latin",
        "taxon_class",
        "body_mass_kg_mean",
        "body_mass_kg_sd",
        "n_individuals",
        "original_interval_min",
        "logger_resolution_C",
        "logger_accuracy_C",
    ]
    # print(species_metadata_df[cols].to_string(index=False))
    return species_metadata_df


def check_sampling(species_data: dict[str, pd.DataFrame]) -> None:
    """Print time-sampling info per species sheet."""
    for name, df_sp in species_data.items():
        if "Hour" not in df_sp.columns:
            print(f"{name:20s}  Hour column not found")
            continue
        hours = pd.to_numeric(df_sp["Hour"], errors="coerce").dropna().values
        if hours.size < 2:
            print(f"{name:20s}  Not enough points to infer dt")
            continue
        dt = float(np.median(np.diff(hours)))
        total_h = float(hours[-1] - hours[0])
        print(
            f"{name:20s}  dt = {dt*60:.1f} min  |  "
            f"total = {total_h:.0f} h  |  N = {len(hours)}"
        )
