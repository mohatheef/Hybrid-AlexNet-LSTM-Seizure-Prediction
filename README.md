# Hybrid AlexNet-LSTM Framework for Early Preictal Seizure Prediction in Pediatric EEG

This repository contains the official implementation of the paper:
**"Hybrid AlexNet-LSTM Framework for Early Preictal Seizure Prediction in Pediatric EEG"**
*Published in NQComp 2026 (International Conference on Next-Gen Quantum and Advanced Computing: Algorithms, Security, and Beyond)*

### Authors
* **Thejan R.** - Manipal Institute of Technology, MAHE, Manipal, India
* **Mohammed Atheef G. A.** - Manipal Institute of Technology, MAHE, Manipal, India
* **Muralidhar Bairy G.** - Manipal Institute of Technology, MAHE, Manipal, India

---

## 📌 Abstract
Epilepsy is a persistent neurological disorder affecting over 50 million people worldwide. Early seizure prediction significantly improves patient safety and quality of life by allowing timely preventive interventions. This paper presents a novel hybrid deep learning architecture combining **AlexNet** (for spatial feature extraction across EEG channels) and **LSTM** (for temporal sequence modeling of EEG dynamics) to predict preictal seizure occurrences **30–60 seconds in advance**. 

Using the **CHB-MIT Scalp EEG pediatric database**, and addressing class imbalance via **SMOTE** and class-weighted training, the model achieves a state-of-the-art **99.36% Accuracy**, **99.93% Sensitivity**, **98.80% Specificity**, **0.9996 AUC**, and an extremely low False Positive Rate (FPR) of **0.012 h⁻¹**.

---

## ⚙️ Preprocessing Pipeline
Raw EEG signals undergo a multistage preprocessing pipeline:
1. **Filtering**: 50 Hz Notch filter (implemented via a second-order IIR filter with quality factor $Q = 30$) to eliminate power-line noise.
2. **Channel Standardization**: Ensures a constant input dimension of 23 channels (padding missing channels with zeros if necessary).
3. **Segmentation & Labeling**: Segmented into short sliding windows of 100 time steps ($\approx 0.39$ seconds at 256 Hz).
4. **Normalization**: Z-score normalization applied to each window.
5. **Class Balancing**: Synthetic Minority Oversampling Technique (SMOTE) applied to training data, combined with class-weighted loss during training ($1.0$ for non-seizure, $5.0$ for seizure).

---

## 🧠 Model Architecture

The proposed network consists of five main phases:
1. **Input Phase**: Expects a 3D tensor of shape `(100, 23, 1)` corresponding to `(time_steps, EEG_channels, depth)`.
2. **Spatial Feature Extraction (AlexNet-inspired)**:
   * **Block 1**: 2D Convolution (64 filters, $3 \times 3$, ReLU) $\rightarrow$ Max Pooling ($2 \times 2$) $\rightarrow$ Batch Normalization
   * **Block 2**: 2D Convolution (128 filters, $3 \times 3$, ReLU) $\rightarrow$ Max Pooling ($2 \times 2$) $\rightarrow$ Batch Normalization
   * **Block 3**: 2D Convolution (256 filters, $3 \times 3$, ReLU) $\rightarrow$ Max Pooling ($2 \times 2$) $\rightarrow$ Batch Normalization
3. **Transition Phase**: Flattening followed by Reshaping to map spatial features back to sequence time steps.
4. **Temporal Modeling (LSTM)**:
   * LSTM Layer 1 (128 units, returning sequences)
   * LSTM Layer 2 (64 units)
5. **Classification (Dense & Dropout)**:
   * Dense Layer (128 units, ReLU) $\rightarrow$ Dropout (0.5)
   * Dense Layer (64 units, ReLU) $\rightarrow$ Dropout (0.5)
   * Output Layer (1 unit, Sigmoid)

---

## 📊 Comparative Performance Results
The performance of the proposed AlexNet-LSTM model is evaluated against CNN, LSTM, and CNN-LSTM models on the CHB-MIT Dataset:

| Method | Accuracy (%) | Sensitivity (%) | Specificity (%) | AUC | FPR ($h^{-1}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CNN | 98.12 | 97.85 | 98.40 | 0.985 | 0.045 |
| LSTM | 98.47 | 98.02 | 98.90 | 0.989 | 0.032 |
| CNN-LSTM | 99.05 | 99.10 | 98.95 | 0.995 | 0.020 |
| **Proposed Work (AlexNet-LSTM)** | **99.36** | **99.93** | **98.80** | **0.9996** | **0.012** |

---

## 📂 Project Structure

```
.
├── README.md                           # This documentation file
├── .gitignore                         # Git exclusion rules
├── requirements.txt                   # Project dependencies
├── src/                               # Main source files
│   ├── hybrid_alexnet_lstm.py         # Main training and evaluation script
│   ├── eeg_visualization.py           # Preictal, ictal, postictal segment plotter
│   ├── spectral_features.py           # PSD, band power & spectral entropy extractor
│   ├── edf_to_csv.py                  # EDF to CSV data converter
│   ├── psd_band_power.py              # Power spectral density calculation
│   └── raw_eeg_plot.py                # Raw EEG visualizer
├── baselines/                         # Baseline comparison models
│   ├── ann_single_patient.py          # Single patient ANN implementation
│   └── ann_multi_patient.py           # Multi-patient ANN implementation
└── notebooks/                         # Jupyter Notebooks
    ├── seizure_prediction.ipynb       # Clean interactive notebook version
    └── seizure_prediction_full.ipynb  # Comprehensive notebook (execution outputs cleared)
```

---

## 🚀 Usage

### 1. Installation
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 2. Preparing the Dataset
The database used is the **CHB-MIT Scalp EEG Database** from PhysioNet. 
* Download the EDF files and summary texts.
* Update the file paths in the scripts (`data_root` and `summary_root`) to point to your local dataset directory.

### 3. Running the Model
Run the hybrid AlexNet-LSTM training and evaluation pipeline:
```bash
python src/hybrid_alexnet_lstm.py
```

### 4. EEG Visualizations
To visualize pre-ictal, ictal, and post-ictal phases from the raw EDF recordings:
```bash
python src/eeg_visualization.py
```
