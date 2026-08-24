# Gene Expression Classifier for Kidney Cancer Subtypes

This project uses machine learning to classify kidney cancer subtypes (**KIRC**, **KICH**, and **KIRP**) from gene expression data.

## Key Links
* **Dataset:** [Link]
* **Reference Pipeline:** [Link]
* **Full Writeup & Results:** [Link to Full Paper/Report]

## Methodology
Gene data was taken and concatenated from three differenet kidney cancer datasets. Gene expression data was then log transformed and filtered with a varaince threshold to preserve only important genes. ANOVA F-test pre-filter and Lasso/Elastic net were used to filter genes into a dominant panel. Remaining data was split into a 70/30 stratified train test split to account for class imblanace between three difference kidney cancer types (KIRC: 537, KIRP: 290, KICH: 66 samples). Data was evaluated with several different classifiers (Decision tree, SVM, ADABoost, KNN).


## Files
data_concat.py - merges TCGA-KIRC/KIRP/KICH cohorts into `data_kidney.csv` / `labels_kidney.csv`
exploratory_analysis.py - visualise class distribution, gene expression distribution, per gene variance, correlation heatmap of top variance genes, boxplots between genes


## Data
Download STAR-TPM gene expression data for TCGA-KIRC, TCGA-KIRP, and TCGA-KICH
from [UCSC Xena](https://xenabrowser.net/datapages/) and place the `.tsv` files
in the project root before running `data_concat.py`.

## Quick Start

git clone https://github.com/souravvvvvvvv/Gene.git
cd Gene
pip install -r requirements.txt
python data_concat.py
python pipeline.py


