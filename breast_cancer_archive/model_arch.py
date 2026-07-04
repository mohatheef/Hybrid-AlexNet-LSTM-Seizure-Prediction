# ============================================
# TCGA BRCA RNA-seq Autoencoder + Classifier Pipeline (Robust)
# ============================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import os

# =========================
# 1. Load preprocessed dataset & labels
# =========================
DATA_PATH = r"C:\sem 2 thejan\Paper\breast cancer dataset\mRNA_preprocessed_TCGA.csv"
LABELS_PATH = r"C:\sem 2 thejan\Paper\breast cancer dataset\mRNA_sample_labels_TCGA.csv"

# Load expression matrix
expr = pd.read_csv(DATA_PATH, index_col=0).T  # samples x genes

# Load labels
labels_df = pd.read_csv(LABELS_PATH, index_col=0)
labels_df['Label'] = labels_df['Label'].map({'Normal':0, 'Tumor':1})
y = labels_df['Label'].values.astype(np.int64)

# Optional: check sample distribution
unique, counts = np.unique(y, return_counts=True)
print("Sample distribution:", dict(zip(unique, counts)))

# Convert to float32
X = expr.values.astype(np.float32)

# =========================
# 2. PyTorch Dataset
# =========================
class GeneDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =========================
# 3. Train-validation split (stratified)
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
train_dataset = GeneDataset(X_train, y_train)
val_dataset = GeneDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# =========================
# 4. Define Autoencoder
# =========================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, bottleneck_dim=50):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder = Autoencoder(input_dim=X.shape[1]).to(device)

# =========================
# 5. Train Autoencoder
# =========================
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
ae_criterion = nn.MSELoss()
EPOCHS_AE = 50

for epoch in range(EPOCHS_AE):
    autoencoder.train()
    total_loss = 0
    for batch_X, _ in train_loader:
        batch_X = batch_X.to(device)
        ae_optimizer.zero_grad()
        recon, _ = autoencoder(batch_X)
        loss = ae_criterion(recon, batch_X)
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
    print(f"Autoencoder Epoch [{epoch+1}/{EPOCHS_AE}], Loss: {total_loss/len(train_dataset):.4f}")

# =========================
# 6. Encode all samples
# =========================
autoencoder.eval()
with torch.no_grad():
    X_encoded, _ = autoencoder(torch.tensor(X).to(device))
    X_encoded = X_encoded.cpu().numpy()

# =========================
# 7. Classifier on bottleneck features
# =========================
class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

classifier = Classifier(input_dim=X_encoded.shape[1]).to(device)
clf_optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
clf_criterion = nn.CrossEntropyLoss()

# Train-validation split for classifier
X_train_enc, X_val_enc, y_train_enc, y_val_enc = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)
train_dataset_enc = GeneDataset(X_train_enc, y_train_enc)
val_dataset_enc = GeneDataset(X_val_enc, y_val_enc)
train_loader_enc = DataLoader(train_dataset_enc, batch_size=32, shuffle=True)
val_loader_enc = DataLoader(val_dataset_enc, batch_size=32, shuffle=False)

# =========================
# 8. Train classifier
# =========================
EPOCHS_CLF = 50
for epoch in range(EPOCHS_CLF):
    classifier.train()
    total_loss = 0
    correct = 0
    for batch_X, batch_y in train_loader_enc:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        clf_optimizer.zero_grad()
        outputs = classifier(batch_X)
        loss = clf_criterion(outputs, batch_y)
        loss.backward()
        clf_optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
        correct += (outputs.argmax(dim=1) == batch_y).sum().item()
    acc = correct / len(train_dataset_enc)
    print(f"Classifier Epoch [{epoch+1}/{EPOCHS_CLF}], Loss: {total_loss/len(train_dataset_enc):.4f}, Accuracy: {acc:.4f}")

# =========================
# 9. Evaluate classifier (robust)
# =========================
classifier.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for batch_X, batch_y in val_loader_enc:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = classifier(batch_X)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

print("\nUnique labels in validation set:", np.unique(all_labels))
print("Unique predictions:", np.unique(all_preds))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

# Accuracy
accuracy = np.sum(all_preds == all_labels) / len(all_labels)
print(f"Accuracy: {accuracy:.4f}")

# Classification report with robust labels
report = classification_report(all_labels, all_preds, labels=[0,1], target_names=['Normal','Tumor'])
print("\nClassification Report:")
print(report)

