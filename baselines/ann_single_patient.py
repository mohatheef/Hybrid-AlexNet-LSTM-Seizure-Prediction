import os
import numpy as np
import mne
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define paths
data_path = r"C:\sem 2 thejan\Paper\edf\chb01"
files = [f for f in os.listdir(data_path) if f.endswith('.edf')]

# Seizure timings (manually extracted from your data)
seizure_info = {
    "chb01_03.edf": (2996, 3036),
    "chb01_04.edf": (1467, 1494),
    "chb01_15.edf": (1732, 1772),
    "chb01_16.edf": (1015, 1066),
    "chb01_18.edf": (1720, 1810),
    "chb01_21.edf": (327, 420),
}

X, y = [], []  # Data and labels

for file in files:
    file_path = os.path.join(data_path, file)
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    data, times = raw[:]
    fs = int(raw.info['sfreq'])  # Sampling frequency
    
    # Create seizure and normal samples
    if file in seizure_info:
        start, end = seizure_info[file]
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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define ANN Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=32)

# Plot training & validation accuracy/loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training vs. Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training vs. Validation Loss')
plt.show()

# Evaluate model
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
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
