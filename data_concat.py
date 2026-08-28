import pandas as pd
import numpy as np

KICH = pd.read_csv("TCGA-KICH.star_tpm.tsv", sep="\t", index_col=0)
tumor_columns = [col for col in KICH.columns if col.split('-')[3].startswith('01')]
KICH_tumor = KICH[tumor_columns]
KICH_tumor = KICH_tumor.T
print(KICH_tumor.head())
print(f'Dimensions of KICH are {KICH_tumor.shape}')

KIRC = pd.read_csv("TCGA-KIRC.star_tpm.tsv", sep="\t", index_col=0)
tumor_columns = [col for col in KIRC.columns if col.split('-')[3].startswith('01')]
KIRC_tumor = KIRC[tumor_columns]
KIRC_tumor = KIRC_tumor.T
print(KIRC_tumor.head())
print(f'Dimensions of KIRCH are {KIRC_tumor.shape}')  

KIRP = pd.read_csv("TCGA-KIRP.star_tpm.tsv", sep="\t", index_col=0)
tumor_columns = [col for col in KIRP.columns if col.split('-')[3].startswith('01')]
KIRP_tumor = KIRP[tumor_columns]
KIRP_tumor = KIRP_tumor.T 
print(KIRP_tumor.head())
print(f'Dimensions of KIRP are {KIRP_tumor.shape}') 

assert list(KICH_tumor.columns) == list(KIRC_tumor.columns) == list(KIRP_tumor.columns), \
    "Gene columns don't match across cohorts — check Ensembl ID versions/order"
# Concatenate samples (rows) together
X_combined = pd.concat([KICH_tumor, KIRC_tumor, KIRP_tumor], axis=0)

# Build matching labels — one per sample, in the same order as X_combined
labels = (
    ['KICH'] * KICH_tumor.shape[0] +
    ['KIRC'] * KIRC_tumor.shape[0] +
    ['KIRP'] * KIRP_tumor.shape[0]
)
y_combined = pd.DataFrame({'Class': labels}, index=X_combined.index)

print(X_combined.shape)
print(y_combined['Class'].value_counts())

# Save out to match your existing pipeline's expected file format
X_combined.to_csv('data_kidney.csv')
y_combined.to_csv('labels_kidney.csv')