import os
import numpy as np
import mne
import tensorflow as tf
from scipy.signal import iirnotch, filtfilt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Flatten, Conv2D, MaxPooling2D, 
                                     BatchNormalization, Input, LSTM, Reshape)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "seizure_data")
summary_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "summaries")

# Constants
max_channels = 23
sequence_length = 100
sampling_rate = 256

# Notch filter
def notch_filter(data, notch_freq=50.0, fs=256, quality_factor=30):
    b, a = iirnotch(notch_freq, quality_factor, fs)
    return filtfilt(b, a, data, axis=1)

# Parse seizure summary
def parse_summary(file_path):
    seizure_info = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    file_name = None
    for line in lines:
        line = line.strip()
        if line.startswith("File Name:"):
            file_name = line.split()[-1]
            seizure_info[file_name] = []
        if line.startswith("Seizure Start Time:"):
            start_time = int(line.split()[-2])
        if line.startswith("Seizure End Time:"):
            end_time = int(line.split()[-2])
            seizure_info[file_name].append((start_time, end_time))
    return seizure_info

# Data preparation
X, y = [], []
for patient_folder in os.listdir(data_root):
    patient_path = os.path.join(data_root, patient_folder)
    summary_file = os.path.join(summary_root, f"{patient_folder}-summary.txt")
    if not os.path.isdir(patient_path) or not os.path.exists(summary_file):
        continue
    seizure_info = parse_summary(summary_file)
    for file in seizure_info.keys():
        file_path = os.path.join(patient_path, file)
        if not os.path.exists(file_path):
            continue
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.rename_channels({ch: f"{ch}_{i}" for i, ch in enumerate(raw.ch_names) if raw.ch_names.count(ch) > 1})
        selected_channels = raw.ch_names[:max_channels]
        raw.pick_channels(selected_channels)
        data, times = raw[:]
        fs = int(raw.info['sfreq']) if 'sfreq' in raw.info else sampling_rate
        data = notch_filter(data, fs=fs)
        if data.shape[0] < max_channels:
            pad_size = max_channels - data.shape[0]
            data = np.pad(data, ((0, pad_size), (0, 0)), mode='constant')
        for start, end in seizure_info[file]:
            seizure_samples = data[:, start * fs: end * fs]
            normal_samples = data[:, :start * fs]
            for i in range(0, seizure_samples.shape[1] - sequence_length, sequence_length):
                X.append(seizure_samples[:, i:i + sequence_length].T)
                y.append(1)
            for i in range(0, normal_samples.shape[1] - sequence_length, sequence_length):
                X.append(normal_samples[:, i:i + sequence_length].T)
                y.append(0)

X = np.array(X)
y = np.array(y)
X = (X - np.mean(X)) / np.std(X)
X = np.expand_dims(X, axis=-1)

# Handle imbalance
X_reshaped = X.reshape(X.shape[0], -1)
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_reshaped, y)
X = X_resampled.reshape(-1, sequence_length, max_channels, 1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_resampled, test_size=0.2, random_state=42)

# Hybrid AlexNet + LSTM Model
def hybrid_alexnet_lstm(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(64, 3, padding='same', activation='relu')(input_layer)
    x = MaxPooling2D(2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D(2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    time_steps = input_shape[0] // 8
    features = x.shape[-1] // time_steps
    x = Reshape((time_steps, features))(x)
    x = LSTM(128, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    return Model(inputs=input_layer, outputs=output_layer)

# Compile and train
input_shape = (sequence_length, max_channels, 1)
model = hybrid_alexnet_lstm(input_shape)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
class_weights = {0: 1, 1: 5}
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=25, batch_size=32, class_weight=class_weights)

# Predictions and Metrics
y_pred_prob = model.predict(X_test)
y_pred = np.round(y_pred_prob)
conf_matrix = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 0] + conf_matrix[0, 1])
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
print(f"Sensitivity: {sensitivity * 100:.2f}%")
print(f"Specificity: {specificity * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"False Positive Rate: {false_positive_rate:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"AUC Score: {roc_auc:.4f}")

# Plot ROC
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Seizure', 'Seizure'], yticklabels=['Non-Seizure', 'Seizure'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Accuracy and Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Plot EEG with Ictal and Postictal Phase
def plot_eeg_with_phases(data, fs, start_time, end_time, channel=0, postictal_duration=10):
    plt.figure(figsize=(15, 4))
    times = np.arange(data.shape[1]) / fs
    eeg_signal = data[channel]
    plt.plot(times, eeg_signal, color='black', linewidth=0.5, label='EEG Signal')
    plt.axvspan(start_time, end_time, color='red', alpha=0.3, label='Ictal Phase')
    plt.axvspan(end_time, end_time + postictal_duration, color='orange', alpha=0.3, label='Postictal Phase')
    plt.title(f'EEG Channel {channel} with Ictal and Postictal Phases')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example EEG visualization for first patient with seizure
for patient_folder in os.listdir(data_root):
    patient_path = os.path.join(data_root, patient_folder)
    summary_file = os.path.join(summary_root, f"{patient_folder}-summary.txt")
    if not os.path.isdir(patient_path) or not os.path.exists(summary_file):
        continue
    seizure_info = parse_summary(summary_file)
    for file_name, events in seizure_info.items():
        file_path = os.path.join(patient_path, file_name)
        if not os.path.exists(file_path):
            continue
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw.pick_channels(raw.ch_names[:max_channels])
        data, times = raw[:]
        fs = int(raw.info['sfreq']) if 'sfreq' in raw.info else sampling_rate
        data = notch_filter(data, fs=fs)
        if len(events) > 0:
            start_sec, end_sec = events[0]
            plot_eeg_with_phases(data, fs, start_sec, end_sec, channel=0, postictal_duration=10)
        break
    break
