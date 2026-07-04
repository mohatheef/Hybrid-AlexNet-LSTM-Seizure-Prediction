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
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns

# **Paths to Data**
data_root = r"C:\sem 2 thejan\Paper\chb-mit-scalp-eeg-database-1.0.0"
summary_root = r"C:\sem 2 thejan\Paper\Thejan seizure\all patient summary"

# **Constants**
max_channels = 23
sequence_length = 100
sampling_rate = 256

# **Notch Filter - Channel-wise**pip install tensorflow-gpu

def notch_filter_per_channel(data, notch_freq=50.0, fs=256, quality_factor=30):
    b, a = iirnotch(notch_freq, quality_factor, fs)
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):  # channel-wise filtering
        filtered_data[i] = filtfilt(b, a, data[i])
    return filtered_data

# **Parse summary file**
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

# **Initialize Data**
X, y = [], []

# **Loop through patients**
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

        # Convert to float32 and filter
        data = data.astype(np.float32)
        data = notch_filter_per_channel(data, fs=fs)

        # Padding if channels < max_channels
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

# **Preprocessing**
X = np.array(X)
y = np.array(y)
X = (X - np.mean(X)) / np.std(X)
X = np.expand_dims(X, axis=-1)

# **SMOTE**
X_reshaped = X.reshape(X.shape[0], -1)
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_reshaped, y)
X = X_resampled.reshape(-1, sequence_length, max_channels, 1)

# **Split**
X_train, X_test, y_train, y_test = train_test_split(X, y_resampled, test_size=0.2, random_state=42)

# **Model Definition**
def hybrid_alexnet_lstm(input_shape):
    input_layer = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=3, padding='same', activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, padding='same')(x)
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

# **Train**
input_shape = (sequence_length, max_channels, 1)
model = hybrid_alexnet_lstm(input_shape)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

class_weights = {0: 1, 1: 5}
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=25, batch_size=32, class_weight=class_weights)

# **Evaluation**
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

# **ROC Curve**
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# **Confusion Matrix**
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Non-Seizure', 'Seizure'],
            yticklabels=['Non-Seizure', 'Seizure'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# **Training Curve**
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation', color='red')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train', color='blue')
plt.plot(history.history['val_loss'], label='Validation', color='red')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()