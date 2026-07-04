import os
import numpy as np
import mne
import tensorflow as tf
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef

# Define paths
data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "seizure_data")
summary_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "summaries", "all_summaries.txt")

# Butterworth Bandpass Filter
def butter_bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=256, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)

# Parse seizure information
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

seizure_info = parse_summary(summary_file)

X, y = [], []  # Data and labels

# Identify the most common 23 channels across all files
channel_counts = {}
for patient in os.listdir(data_root):
    patient_path = os.path.join(data_root, patient)
    if not os.path.isdir(patient_path):
        continue
    files = [f for f in os.listdir(patient_path) if f.endswith('.edf')]
    
    for file in files:
        file_path = os.path.join(patient_path, file)
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        for ch in raw.ch_names:
            channel_counts[ch] = channel_counts.get(ch, 0) + 1

# Select the 23 most common channels
sorted_channels = sorted(channel_counts, key=channel_counts.get, reverse=True)
standard_channels = sorted_channels[:23]

# Process EEG data
for patient in os.listdir(data_root):
    patient_path = os.path.join(data_root, patient)
    if not os.path.isdir(patient_path):
        continue
    files = [f for f in os.listdir(patient_path) if f.endswith('.edf')]
    
    for file in files:
        file_path = os.path.join(patient_path, file)
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # Adjust channels: Keep only the most common 23 channels
        available_channels = [ch for ch in raw.ch_names if ch in standard_channels]
        raw.pick_channels(available_channels, ordered=True, verbose=False)
        
        data, times = raw[:]
        fs = int(raw.info['sfreq'])  # Sampling frequency
        
        # Apply Butterworth Filter
        data = butter_bandpass_filter(data, fs=fs)
        
        # Create seizure and normal samples
        if file in seizure_info and seizure_info[file]:
            for start, end in seizure_info[file]:
                seizure_samples = data[:, start * fs: end * fs]
                normal_samples = data[:, :start * fs]  # Non-seizure samples
                
                X.append(seizure_samples.T)
                y.append(np.ones(len(seizure_samples.T)))  # Label as seizure
                
                X.append(normal_samples.T)
                y.append(np.zeros(len(normal_samples.T)))  # Label as normal

# Convert to numpy arrays
X = np.vstack(X)
y = np.concatenate(y)

# Normalize data
X = (X - np.mean(X)) / np.std(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define ANN Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

# Compute Metrics
conf_matrix = confusion_matrix(y_test, y_pred)
sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
false_positive_rate = conf_matrix[0, 1] / (conf_matrix[0, 0] + conf_matrix[0, 1])
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
mcc = matthews_corrcoef(y_test, y_pred)

# Print Results
print(f"Accuracy: {acc * 100:.2f}%")
print(f"Sensitivity: {sensitivity * 100:.2f}%")
print(f"Specificity: {specificity * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"False Positive Rate (per hr): {false_positive_rate:.4f}")
print(f"F1 Score: {f1_score:.4f}")
print(f"MCC: {mcc:.4f}")

# Generate classification report
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs. Validation Accuracy')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training vs. Validation Loss')
plt.show()