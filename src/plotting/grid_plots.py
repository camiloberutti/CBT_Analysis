"""Grid and cross-species plotting functions that combine PSD computation and plotting."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.psd.maczak import compute_maczak_psd
from src.psd.welch import compute_welch_psd
from src.psd.irasa import compute_irasa_psd
from src.parametrization.pwlf_fit import compute_pwlf_log
from src.plotting.psd_plots import plot_psd_single, plot_psd_with_local_beta


# ── Species PSD grid (3×3) ──────────────────────────────────────────────────

def plot_species_grid(
    species_data_clean,
    df_species_meta,
    method="maczak",
    *,
    # ── Maczak-specific ──
    bins_per_decade=100,
    # ── Welch-specific ──
    nperseg=None,
    noverlap=None,
    window="hann",
    # ── IRASA-specific ──
    f_range=None,
    hset=None,
    irasa_thresh=None,
):
    """
    Plots a 3×3 grid of PSDs for different species with independent Y-axis scaling.

    Parameters
    ----------
    method          : 'maczak', 'welch', or 'irasa'.
    bins_per_decade : (maczak) log-binning resolution.
    nperseg         : (welch) segment length.
    noverlap        : (welch) overlap.
    window          : (welch) window function.
    f_range         : (irasa) frequency range.
    hset            : (irasa) resampling factors.
    irasa_thresh    : (irasa) periodic threshold.
    """
    species_keys = list(species_data_clean.keys())
    n_species = len(species_keys)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(
        16, 12), sharex=True, sharey=False)
    axes_flat = axes.flatten()

    for i, specie_name in enumerate(species_keys):
        ax = axes_flat[i]

        # Get and clean data
        df_sp = species_data_clean[specie_name]
        signal_cols = [c for c in df_sp.columns if c != "Hour"]
        signals = [pd.to_numeric(df_sp[c], errors="coerce").dropna(
        ).to_numpy() for c in signal_cols]
        signals = [s for s in signals if s.size > 3]

        dt_min = df_species_meta.loc[df_species_meta["sheet_name"]
                                     == specie_name, "original_interval_min"].iloc[0]
        fs_specie = float(1.0 / (dt_min * 60.0))

        # Compute PSD based on selected method
        aperiodic = None
        periodic = None

        if method.lower() == "maczak":
            freqs, PSD = compute_maczak_psd(
                signals, fs_specie, bins_per_decade=bins_per_decade)
            plot_nperseg = None

        elif method.lower() == "welch":
            actual_nperseg = nperseg if nperseg is not None else min(
                len(s) for s in signals)
            plot_nperseg = actual_nperseg

            all_psds = []
            freqs = None
            for s in signals:
                f, p = compute_welch_psd(
                    s, fs_specie, nperseg=actual_nperseg, noverlap=noverlap, window=window)
                if freqs is None:
                    freqs = f
                all_psds.append(p)
            PSD = np.mean(all_psds, axis=0)

        elif method.lower() == "irasa":
            plot_nperseg = None
            all_psds = []
            all_aperiodic = []
            all_periodic = []
            freqs = None

            for s in signals:
                try:
                    f, aper, osc = compute_irasa_psd(
                        s, fs_specie,
                        f_range=f_range,
                        hset=hset,
                        thresh=irasa_thresh,
                    )
                    if freqs is None:
                        freqs = f
                    total_psd = aper + osc
                    all_psds.append(total_psd)
                    all_aperiodic.append(aper)
                    all_periodic.append(osc)
                except Exception as e:
                    print(f"Skipping signal for IRASA in {specie_name}: {e}")

            if not all_psds:
                print(f"No valid IRASA results for {specie_name}.")
                continue

            PSD = np.mean(all_psds, axis=0)
            aperiodic = np.mean(all_aperiodic, axis=0)
            periodic = np.mean(all_periodic, axis=0)

        else:
            raise ValueError(
                "Method must be either 'maczak', 'welch', or 'irasa'")

        # Plot on the specific subplot (ax)
        plot_psd_single(
            freqs,
            PSD,
            fs_specie,
            specie=specie_name,
            n_individuals=len(signals),
            nperseg=plot_nperseg,
            ax=ax,
            show_markers=True,
            aperiodic=aperiodic,
            periodic=periodic
        )

        # --- Clean up labels and legends ---
        # Keep legend only for the first subplot, remove for others
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
        else:
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        # Clear the default labels applied by plot_psd_single
        ax.set_xlabel("")
        ax.set_ylabel("")

        # Only apply the Y-axis name if it's the far-left column
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("PSD (°C² / Hz)", fontsize=12)

        # Only apply the X-axis name if it's the bottom row
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel("Frequency (Hz)", fontsize=12)

    # Hide any empty subplots
    for j in range(n_species, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # Add a master title and render
    fig.suptitle(
        f"Power Spectral Density across Species ({method.capitalize()} Method)", fontsize=18, fontweight='bold', y=1.02)

    plt.tight_layout(h_pad=1.5, w_pad=1.5)
    plt.show()


# ── Species beta grid (3×3) ─────────────────────────────────────────────────

def plot_species_beta_grid(species_data_clean, df_species_meta, method="maczak", fit_fmin=None, fit_fmax=None):
    """
    Plots a 3×3 grid of PSD + Local Beta for all species.
    """
    species_keys = list(species_data_clean.keys())
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 14), sharex=True)
    axes_flat = axes.flatten()

    for i, specie_name in enumerate(species_keys):
        ax = axes_flat[i]

        df_sp = species_data_clean[specie_name]
        signal_cols = [c for c in df_sp.columns if c != "Hour"]
        signals = [pd.to_numeric(df_sp[c], errors="coerce").dropna(
        ).to_numpy() for c in signal_cols]
        signals = [s for s in signals if s.size > 3]

        dt_min = df_species_meta.loc[df_species_meta["sheet_name"]
                                     == specie_name, "original_interval_min"].iloc[0]
        fs_specie = float(1.0 / (dt_min * 60.0))

        # Simple PSD fetch (using Maczak as default for grid)
        if method == "maczak":
            freqs, PSD = compute_maczak_psd(
                signals, fs_specie, bins_per_decade=100)
        else:
            # Placeholder for other methods if needed
            continue

        plot_psd_with_local_beta(freqs, PSD, fs_specie, specie=specie_name,
                                 fit_fmin=fit_fmin, fit_fmax=fit_fmax, ax=ax)

        # Clean up labels for grid view
        if i % 3 != 0:
            ax.set_ylabel("")
        if i < 6:
            ax.set_xlabel("")

    plt.suptitle(
        f"PSD and Local Beta across Species ({method.capitalize()})", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()


# ── Parameters vs weight ─────────────────────────────────────────────────────

def parameters_vs_weight_species(data, df_species_meta, psd_method='maczak', n_lines=3, **kwargs):
    """
    Computes PSD fit parameters across all species and plots them against body weight.

    Parameters
    ----------
    psd_method : str — 'maczak', 'welch', or 'irasa'.
    n_lines    : int — number of line segments for piecewise linear regression.
    **kwargs   : method-specific arguments:
        - bins_per_decade : int (maczak, default 100)
        - nperseg         : int or None (welch)
        - noverlap        : int or None (welch)
        - window          : str (welch, default 'hann')
        - f_range         : tuple or None (irasa)
        - hset            : array or None (irasa)
        - irasa_thresh    : float or None (irasa)
    """
    bins_per_decade = kwargs.get('bins_per_decade', 100)
    nperseg = kwargs.get('nperseg', None)
    noverlap = kwargs.get('noverlap', None)
    window = kwargs.get('window', 'hann')
    f_range = kwargs.get('f_range', None)
    hset = kwargs.get('hset', None)
    irasa_thresh = kwargs.get('irasa_thresh', None)

    pwlf_results = []

    print(f"Method: {psd_method}, n_lines: {n_lines})...")

    # 2. Iterate through species
    for specie_name in data.keys():
        try:
            df_sp = data[specie_name]
            signal_cols = [c for c in df_sp.columns if c != "Hour"]
            signals = [pd.to_numeric(df_sp[c], errors="coerce").dropna(
            ).to_numpy() for c in signal_cols]
            signals = [s for s in signals if s.size > 10]

            if not signals:
                continue

            # Get sampling frequency
            dt_min = df_species_meta.loc[df_species_meta["sheet_name"]
                                         == specie_name, "original_interval_min"].iloc[0]
            fs_specie = float(1.0 / (dt_min * 60.0))

            # ── Compute PSD based on method ──
            if psd_method.lower() == 'maczak':
                freqs, PSD = compute_maczak_psd(
                    signals, fs_specie, bins_per_decade=bins_per_decade)

            elif psd_method.lower() == 'welch':
                actual_nperseg = nperseg if nperseg is not None else min(
                    len(s) for s in signals)
                all_psds = []
                freqs = None
                for s in signals:
                    f, p = compute_welch_psd(
                        s, fs_specie, nperseg=actual_nperseg, noverlap=noverlap, window=window)
                    if freqs is None:
                        freqs = f
                    all_psds.append(p)
                PSD = np.mean(all_psds, axis=0)

            elif psd_method.lower() == 'irasa':
                all_aperiodic = []
                freqs = None
                for s in signals:
                    try:
                        f, aper, _ = compute_irasa_psd(
                            s, fs_specie,
                            f_range=f_range,
                            hset=hset,
                            thresh=irasa_thresh,
                        )
                        if freqs is None:
                            freqs = f
                        all_aperiodic.append(aper)
                    except Exception:
                        continue
                if not all_aperiodic:
                    continue
                PSD = np.mean(all_aperiodic, axis=0)
            else:
                print(f"Method '{psd_method}' not recognized.")
                return

            # ── Fit PWLF ──
            my_pwlf = compute_pwlf_log(freqs, PSD, n_lines=n_lines)

            # Extract results
            res = {'sheet_name': specie_name}
            for i, slope in enumerate(my_pwlf.slopes):
                res[f'slope_{i+1}'] = slope

            # Internal breaks are between segments (n_lines - 1 breaks)
            breaks_hz = 10**my_pwlf.fit_breaks
            for i in range(1, len(breaks_hz) - 1):
                res[f'break_{i}_hz'] = breaks_hz[i]

            pwlf_results.append(res)
        except Exception as e:
            print(f"  Error on {specie_name}: {e}")

    # 3. Merge with Metadata
    if not pwlf_results:
        print("No results to plot.")
        return

    df_pwlf = pd.DataFrame(pwlf_results)
    df_final = df_pwlf.merge(df_species_meta, on='sheet_name')

    # 4. Dynamic Plotting
    # Identify parameter columns (slopes and breaks)
    param_cols = [c for c in df_pwlf.columns if c != 'sheet_name']
    n_params = len(param_cols)
    cols = 2
    rows = (n_params + 1) // 2

    fig, axs = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    fig.suptitle(f'PSD Parameters ({n_lines} lines) vs Body Mass', fontsize=18)
    axs = axs.flatten()

    for i, col in enumerate(param_cols):
        ax = axs[i]
        ax.scatter(df_final['body_mass_kg_mean'],
                   df_final[col], color='steelblue', alpha=0.7)

        # Annotate species name
        for j, txt in enumerate(df_final['sheet_name']):
            ax.annotate(txt, (df_final['body_mass_kg_mean'].iloc[j],
                        df_final[col].iloc[j]), fontsize=8, alpha=0.8)

        ax.set_xscale('log')
        if 'break' in col:
            ax.set_yscale('log')
            ax.set_ylabel('Frequency (Hz)')
        else:
            ax.set_ylabel('Slope Value')

        ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
        ax.set_xlabel('Mean Body Mass (kg)')
        ax.grid(True, which="both", ls="-", alpha=0.2)

    # Remove empty subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
