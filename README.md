# CBT Spectral Analysis

Spectral analysis of core body temperature (CBT) across 10 species (human + 9 from Maloney et al. 2024).

## Repository structure

```
Analysis/
├── notebooks/
│   └── Analysis.ipynb        # Main analysis notebook
├── src/
│   ├── __init__.py
│   ├── loading.py            # Data loading (CSV, Excel)
│   ├── preprocessing.py      # NaN interpolation, outlier cleaning
│   ├── psd.py                # PSD computation (Welch)
│   ├── plotting.py           # All visualisation functions
│   └── species_metadata.py   # Species metadata from the paper
├── data/
│   ├── paper/                # Public dataset (Maloney et al. 2024)
│   │   ├── Five_days_of_Tc_data_in_nine_species.xlsx
│   │   └── README.md
│   └── group/                # Private sensor data (git-ignored)
│       └── *.csv
├── .gitignore
├── README.md
└── requirements.txt
```

## Data sources

| Dataset       | Source                                                                       | Access                    |
| ------------- | ---------------------------------------------------------------------------- | ------------------------- |
| 9-species CBT | [Dryad doi:10.5061/dryad.1g1jwsv46](https://doi.org/10.5061/dryad.1g1jwsv46) | Public                    |
| Human CBT     | greenTEG CORE sensor                                                         | **Private** (git-ignored) |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Then open `notebooks/Analysis.ipynb` and run all cells.

## License

The analysis code is released under the [MIT License](LICENSE).  
The dataset in `data/paper/` originates from Maloney et al. 2024 and is
distributed under the Dryad default license (CC0 1.0 / CC-BY).
Please cite the original paper if you use the data.

## References

- Goh, G., Vesterdorf, K., Fuller, A., Blache, D., & Maloney, S. K. (2024).
  *Optimal sampling interval for characterisation of the circadian rhythm of body temperature
  in homeothermic animals.* Ecology and Evolution, 14(4), e11243.
