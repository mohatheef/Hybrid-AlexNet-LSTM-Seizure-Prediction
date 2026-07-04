import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ---------------- Paths ----------------
data_root = r"C:\sem 2 thejan\Paper\chb-mit-scalp-eeg-database-1.0.0"
summary_root = r"C:\sem 2 thejan\Paper\Thejan seizure\all patient summary"
os.makedirs(summary_root, exist_ok=True)

# ---------------- EEG Bands ----------------
bands = {
    "Delta (0.5–4 Hz)": (0.5, 4),
    "Theta (4–8 Hz)": (4, 8),
    "Alpha (8–13 Hz)": (8, 13),
    "Beta (13–30 Hz)": (13, 30),
    "Gamma (30–40 Hz)": (30, 40),
}

# ---------------- Helper Functions ----------------
def clean_and_apply_montage(raw):
    """Clean EEG channels and apply standard 10-20 montage."""
    # Drop flat channels
    flat_chs = [ch for ch in raw.ch_names if np.allclose(raw.get_data(picks=ch), 0)]
    if flat_chs:
        print(f"      Dropping flat channels: {flat_chs}")
        raw.drop_channels(flat_chs)

    # Convert bipolar to monopolar (take first electrode)
    clean_ch_names = [ch.split('-')[0] for ch in raw.ch_names]

    # Ensure unique names
    unique_names = []
    for name in clean_ch_names:
        if name in unique_names:
            count = unique_names.count(name)
            unique_names.append(f"{name}_{count+1}")
        else:
            unique_names.append(name)
    raw.rename_channels(dict(zip(raw.ch_names, unique_names)))

    # Apply montage
    montage = mne.channels.make_standard_montage("standard_1020")
    available_chs = [ch for ch in raw.ch_names if ch in montage.ch_names]
    raw.pick(available_chs)
    raw.set_montage(montage, on_missing='ignore')

    return raw

def compute_spectral_features(psds, freqs, bands, ch_names):
    """Compute extended spectral features per channel."""
    features = {}
    total_power = psds.sum(axis=1, keepdims=True)
    for band, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = psds[:, mask].mean(axis=1)
        rel_power = band_power / (total_power.mean(axis=1) + 1e-10)
        peak_freqs = freqs[mask][psds[:, mask].argmax(axis=1)]
        psd_norm = psds[:, mask] / psds[:, mask].sum(axis=1, keepdims=True)
        spec_entropy = entropy(psd_norm, base=2, axis=1)
        features[f"{band}_power"] = band_power
        features[f"{band}_rel_power"] = rel_power
        features[f"{band}_peak_freq"] = peak_freqs
        features[f"{band}_entropy"] = spec_entropy
    return pd.DataFrame(features, index=ch_names)

def visualize_spectral_analysis(raw, psds, freqs, band_power, spectral_features, bands, patient_out, file_prefix):
    """Visualize PSD, band power, and topographic maps."""
    # ---------- PSD ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, ch in enumerate(raw.ch_names):
        ax.semilogy(freqs, psds[i], label=ch)
    ax.set_title("EEG Power Spectral Density (PSD)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (uV²/Hz)")
    ax.set_xlim(0.5, 40)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc="upper right", fontsize=6, ncol=4)
    plt.tight_layout()
    plt.savefig(f"{patient_out}/{file_prefix}_PSD.png", dpi=300)
    plt.close()

    # ---------- Band Power Bar Chart ----------
    fig, ax = plt.subplots(figsize=(10, 5))
    band_power.mean().plot(kind="bar", ax=ax, color="skyblue", rot=45)
    ax.set_title("Mean EEG Band Power Across Channels")
    ax.set_ylabel("Power (uV²/Hz)")
    ax.set_xlabel("Frequency Band")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{patient_out}/{file_prefix}_BandPower.png", dpi=300)
    plt.close()

    # ---------- Topographic Maps ----------
    fig, axes = plt.subplots(1, len(bands), figsize=(20, 4))
    for ax, (band, values) in zip(axes, band_power.items()):
        mne.viz.plot_topomap(values, raw.info, axes=ax, show=False, cmap="viridis")
        ax.set_title(band)
    plt.tight_layout()
    plt.savefig(f"{patient_out}/{file_prefix}_Topomap_Power.png", dpi=300)
    plt.close()

    # Relative power and entropy topomap
    spectral_features_to_plot = []
    for band in bands.keys():
        spectral_features_to_plot.append((f"{band}_rel_power", f"{band} Relative Power"))
        spectral_features_to_plot.append((f"{band}_entropy", f"{band} Spectral Entropy"))

    fig, axes = plt.subplots(2, len(bands), figsize=(22, 8))
    for idx, (feature_col, title) in enumerate(spectral_features_to_plot):
        row = idx // len(bands)
        col = idx % len(bands)
        ax = axes[row, col]
        values = spectral_features[feature_col].values
        im, _ = mne.viz.plot_topomap(values, raw.info, axes=ax, show=False, cmap='coolwarm')
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{patient_out}/{file_prefix}_Topomap_Features.png", dpi=400)
    plt.close()

# ---------------- Main Processing Loop ----------------
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

            # Save PSD CSV
            df_psd = pd.DataFrame(psds.T, columns=raw.ch_names)
            df_psd.insert(0, "Frequency_Hz", freqs)
            df_psd.to_csv(os.path.join(patient_out, file.replace(".edf", "_psd.csv")), index=False)

            # Compute band power
            band_power = {}
            for band, (fmin, fmax) in bands.items():
                mask = (freqs >= fmin) & (freqs <= fmax)
                band_power[band] = psds[:, mask].mean(axis=1)
            band_df = pd.DataFrame(band_power, index=raw.ch_names)
            band_df.to_csv(os.path.join(patient_out, file.replace(".edf", "_bands.csv")))

            # Compute extended spectral features
            spectral_features_df = compute_spectral_features(psds, freqs, bands, raw.ch_names)
            spectral_features_df.to_csv(os.path.join(patient_out, file.replace(".edf", "_spectral_features.csv")))

            # Visualize spectral analysis
            visualize_spectral_analysis(raw, psds, freqs, band_df, spectral_features_df, bands, patient_out, file.replace(".edf", ""))

            print(f"      ✔ Processed {file}")

        except Exception as e:
            print(f"      ⚠ Error processing {file}: {e}")
