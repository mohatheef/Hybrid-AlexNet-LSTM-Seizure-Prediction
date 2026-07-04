import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
data_root = r"C:\sem 2 thejan\Paper\chb-mit-scalp-eeg-database-1.0.0"
summary_root = r"C:\sem 2 thejan\Paper\Thejan seizure\all patient summary"
os.makedirs(summary_root, exist_ok=True)

# Define EEG bands
bands = {
    "Delta (0.5–4 Hz)": (0.5, 4),
    "Theta (4–8 Hz)": (4, 8),
    "Alpha (8–13 Hz)": (8, 13),
    "Beta (13–30 Hz)": (13, 30),
    "Gamma (30–40 Hz)": (30, 40),
}

def clean_and_apply_montage(raw):
    """
    Clean EEG channels: remove flat channels, rename duplicates, keep only montage channels.
    Apply standard 10-20 montage.
    """
    # 1. Drop empty/flat channels
    flat_chs = [ch for ch in raw.ch_names if np.allclose(raw.get_data(picks=ch), 0)]
    if flat_chs:
        print(f"      Dropping flat channels: {flat_chs}")
        raw.drop_channels(flat_chs)

    # 2. Convert bipolar to monopolar (take first electrode)
    clean_ch_names = [ch.split('-')[0] for ch in raw.ch_names]

    # 3. Ensure uniqueness
    unique_names = []
    for name in clean_ch_names:
        if name in unique_names:
            count = unique_names.count(name)
            unique_names.append(f"{name}_{count+1}")
        else:
            unique_names.append(name)
    rename_dict = dict(zip(raw.ch_names, unique_names))
    raw.rename_channels(rename_dict)

    # 4. Apply montage (only for channels present)
    montage = mne.channels.make_standard_montage("standard_1020")
    available_chs = [ch for ch in raw.ch_names if ch in montage.ch_names]
    raw.pick(available_chs)
    raw.set_montage(montage, on_missing='ignore')

    return raw

# ----------------- Main Processing Loop -----------------
for patient in os.listdir(data_root):
    patient_path = os.path.join(data_root, patient)
    if not os.path.isdir(patient_path):
        continue

    print(f"\n=== Processing patient: {patient} ===")
    patient_out = os.path.join(summary_root, patient)
    os.makedirs(patient_out, exist_ok=True)

    for file in os.listdir(patient_path):
        if not file.endswith(".edf"):
            continue
        edf_path = os.path.join(patient_path, file)
        print(f"   Loading {file}")

        try:
            # Load EDF
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.pick("eeg")

            # Clean channels and apply montage
            raw = clean_and_apply_montage(raw)

            if len(raw.ch_names) == 0:
                print(f"      ⚠ No valid channels remaining after cleaning. Skipping {file}.")
                continue

            # Compute PSD
            psd = raw.compute_psd(fmin=0.5, fmax=40, n_fft=1024, verbose=False)
            psds, freqs = psd.get_data(return_freqs=True)

            # ---------- Save Raw PSD to CSV ----------
            df_psd = pd.DataFrame(psds.T, columns=raw.ch_names)
            df_psd.insert(0, "Frequency_Hz", freqs)
            csv_path = os.path.join(patient_out, file.replace(".edf", "_psd.csv"))
            df_psd.to_csv(csv_path, index=False)

            # ---------- Compute Band Power ----------
            band_power = {}
            for band, (fmin, fmax) in bands.items():
                mask = (freqs >= fmin) & (freqs <= fmax)
                band_power[band] = psds[:, mask].mean(axis=1)

            band_df = pd.DataFrame(band_power, index=raw.ch_names)
            band_csv = os.path.join(patient_out, file.replace(".edf", "_bands.csv"))
            band_df.to_csv(band_csv)

            # ---------- Visualization ----------
            fig, axs = plt.subplots(2, 1, figsize=(12, 10))

            # PSD for all channels
            for i, ch in enumerate(raw.ch_names):
                axs[0].semilogy(freqs, psds[i], label=ch)
            axs[0].set_title("PSD (all channels)")
            axs[0].set_xlabel("Frequency (Hz)")
            axs[0].set_ylabel("PSD (uV^2/Hz)")
            axs[0].legend(loc="upper right", fontsize=7, ncol=4)

            # Band Power Bar Chart
            band_df.mean().plot(kind="bar", ax=axs[1], color="skyblue", rot=45)
            axs[1].set_title("EEG Band Power (averaged across channels)")
            axs[1].set_ylabel("Mean PSD Power")

            plt.tight_layout()
            plot_path = os.path.join(patient_out, file.replace(".edf", "_plots.png"))
            plt.savefig(plot_path)
            plt.close()

            # ---------- Topographic Maps ----------
            fig, axes = plt.subplots(1, len(bands), figsize=(18, 4))
            for ax, (band, values) in zip(axes, band_power.items()):
                mne.viz.plot_topomap(values, raw.info, axes=ax, show=False, cmap="viridis")
                ax.set_title(band)

            topo_path = os.path.join(patient_out, file.replace(".edf", "_topomap.png"))
            plt.savefig(topo_path, dpi=300)
            plt.close()
            
            # ---------- Topographic Maps (Heatmap) ----------
            fig, axes = plt.subplots(1, len(bands), figsize=(18, 4))
            for ax, (band, values) in zip(axes, band_power.items()):
                im, cn = mne.viz.plot_topomap(
                    values, raw.info, axes=ax, show=False,
                    cmap='hot',    # Heatmap style
                    contours=0     # Remove contour lines
                )
                ax.set_title(band)
            # Add colorbar
            fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05)
            topo_path = os.path.join(patient_out, file.replace(".edf", "_topomap_heatmap.png"))
            plt.savefig(topo_path, dpi=300)
            plt.close()
            
            
            print(f"      ✔ Saved PSD CSV, band CSV, plots, and topomap for {file}")

        except Exception as e:
            print(f"      ⚠ Error processing {file}: {e}")
