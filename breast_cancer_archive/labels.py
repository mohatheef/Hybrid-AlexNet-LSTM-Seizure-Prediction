# ============================================
# RNA-seq Preprocessing Pipeline (TCGA BRCA)
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. Load dataset
# --------------------------------------------
file_path = r"C:\sem 2 thejan\Paper\breast cancer\csv\data_mrna_seq_v2_rsem.csv"

df = pd.read_csv(file_path)
print("Dataset loaded:", df.shape)

# --------------------------------------------
# 2. Separate annotation & expression matrix
# --------------------------------------------
annotation_cols = []
if "Hugo_Symbol" in df.columns:
    annotation_cols.append("Hugo_Symbol")
if "Entrez_Gene_Id" in df.columns:
    annotation_cols.append("Entrez_Gene_Id")

expr = df.drop(columns=annotation_cols)
print("Expression matrix shape:", expr.shape)

# --------------------------------------------
# 3. Remove all-zero genes
# --------------------------------------------
non_zero_mask = (expr != 0).any(axis=1)
expr = expr.loc[non_zero_mask]

if annotation_cols:
    df = df.loc[non_zero_mask, annotation_cols]

print("After removing all-zero genes:", expr.shape)

# --------------------------------------------
# 4. Low-expression filtering
# --------------------------------------------
min_samples = int(0.20 * expr.shape[1])
expressed_mask = (expr > 1).sum(axis=1) >= min_samples
expr = expr.loc[expressed_mask]
if annotation_cols:
    df = df.loc[expressed_mask, :]
print("After low-expression filtering:", expr.shape)

# --------------------------------------------
# 5. Log2 transformation
# --------------------------------------------
expr_log = np.log2(expr + 1)
print("Log2 transformation applied")

# --------------------------------------------
# 5a. Visual example: raw vs log2
# --------------------------------------------
sample_example = expr.iloc[0:5, 0:5]  # first 5 genes & first 5 samples
sample_example_log = expr_log.iloc[0:5, 0:5]

print("\nRaw values (example):")
print(sample_example)
print("\nLog2-transformed values (example):")
print(sample_example_log)

# Optional: simple plot
plt.figure(figsize=(6,4))
plt.plot(sample_example.values.flatten(), marker='o', linestyle='', label='Raw')
plt.plot(sample_example_log.values.flatten(), marker='x', linestyle='', label='Log2')
plt.xlabel('Values')
plt.ylabel('Expression')
plt.title('Raw vs Log2-transformed example')
plt.legend()
plt.show()

# --------------------------------------------
# 6. Handle duplicate genes
# --------------------------------------------
if "Hugo_Symbol" in annotation_cols:
    expr_log["Hugo_Symbol"] = df["Hugo_Symbol"].values
    expr_log["mean_expr"] = expr_log.drop(columns=["Hugo_Symbol"]).mean(axis=1)
    expr_log = (
        expr_log.sort_values("mean_expr", ascending=False)
                .drop_duplicates(subset="Hugo_Symbol")
                .drop(columns="mean_expr")
    )
    expr_log = expr_log.set_index("Hugo_Symbol")
    print("Duplicate genes resolved")
else:
    expr_log.index = expr_log.index.astype(str)

# --------------------------------------------
# 7. Scaling (Z-score)
# --------------------------------------------
scaler = StandardScaler()
expr_scaled = scaler.fit_transform(expr_log)
expr_scaled = pd.DataFrame(expr_scaled, index=expr_log.index, columns=expr_log.columns)
print("Z-score scaling applied")

# --------------------------------------------
# 7a. Visual example: log2 vs z-score
# --------------------------------------------
sample_example_scaled = expr_scaled.iloc[0:5, 0:5]
print("\nZ-score scaled values (example):")
print(sample_example_scaled)

# --------------------------------------------
# 8. Add Tumor/Normal labels
# --------------------------------------------
def get_sample_type(barcode):
    try:
        code = barcode.split('-')[3][:2]
        if code == '01':
            return 'Tumor'
        elif code == '11':
            return 'Normal'
        else:
            return 'Other'
    except IndexError:
        return 'Unknown'

sample_labels = {col: get_sample_type(col) for col in expr_scaled.columns}
labels_df = pd.DataFrame.from_dict(sample_labels, orient='index', columns=['Label'])
labels_df.index.name = 'Sample_ID'

labels_output_path = r"C:\sem 2 thejan\Paper\breast cancer dataset\mRNA_sample_labels_TCGA.csv"
labels_df.to_csv(labels_output_path)
print("Sample labels saved to:", labels_output_path)

# --------------------------------------------
# 9. Save processed dataset
# --------------------------------------------
output_path = r"C:\sem 2 thejan\Paper\breast cancer dataset\mRNA_preprocessed_TCGA.csv"
expr_scaled.to_csv(output_path)
print("Preprocessing completed successfully!")
print("Final dataset shape:", expr_scaled.shape)
print("Processed expression saved to:", output_path)
