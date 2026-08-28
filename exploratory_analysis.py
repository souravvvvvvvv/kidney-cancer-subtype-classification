import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

X = pd.read_csv('data_kidney.csv', index_col=0)
y = pd.read_csv('labels_kidney.csv')

'''
# --- 1. Class distribution ---
plt.figure(figsize=(6,4))
y['Class'].value_counts().plot(kind='bar', color=['#4C72B0','#DD8452','#55A868'])
plt.title('Sample count per kidney cancer subtype')
plt.ylabel('Number of samples')
plt.xticks(rotation=0)
plt.show()

# --- 2. Distribution of expression values overall ---
plt.figure(figsize=(6,4))
plt.hist(X.values.flatten(), bins=100)
plt.xlabel('TPM expression value')
plt.ylabel('Frequency')
plt.title('Distribution of gene expression values (raw)')
plt.show()
'''

X = np.log1p(X)

# --- 3. Per-gene variance — which genes vary the most across samples ---
gene_variances = X.var().sort_values(ascending=False)
plt.figure(figsize=(6,4))
plt.plot(gene_variances.values)
plt.yscale('log')
plt.xlabel('Gene rank (sorted by variance)')
plt.ylabel('Variance')
plt.title('Full variance distribution')
plt.legend()
plt.show()
'''
# --- 4. Correlation heatmap of top variable genes ---
top_var_genes = gene_variances.head(50).index
corr = X[top_var_genes].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Correlation among top 50 most variable genes')
plt.show()


# --- 5. Boxplots: how a few top-variance genes differ across classes ---
top5_genes = gene_variances.head(5).index
X_with_class = X[top5_genes].copy()
X_with_class['Class'] = y['Class'].values

fig, axes = plt.subplots(1, 5, figsize=(20,4))
for i, gene in enumerate(top5_genes):
    sns.boxplot(data=X_with_class, x='Class', y=gene, ax=axes[i])
    axes[i].set_title(gene, fontsize=9)
plt.tight_layout()
plt.show()
'''