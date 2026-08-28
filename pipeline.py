import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

#Importing the concatenated dataset

X = pd.read_csv('data_kidney.csv', index_col=0)
y = pd.read_csv('labels_kidney.csv')

X = np.log1p(X)

gene_variances = X.var().sort_values(ascending=False)

#Removing all genes that have a variance lower that 0.001 across all samples
kept = (gene_variances > 0.001)
kept_gene_names = gene_variances[kept].index
X_reduced = X[kept_gene_names] 
print(f"Kept {X_reduced.shape[1]} of {X.shape[1]} genes")  

y_target = y['Class']

#Train test split 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y_target, test_size=0.30, stratify=y_target, random_state=42
)


scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

#Running the ANOVA f statistic test to obtain two different cases of 2000 and 10000 for our selectKbest algorithm
f_scores, p_values = f_classif(X_train_s, y_train)
f_scores_sorted = pd.Series(f_scores, index=X_train.columns).sort_values(ascending=False)


plt.plot(f_scores_sorted.values[:50000])
plt.xlabel('Gene rank (by F-score)')
plt.ylabel('F-statistic')
plt.show()


#Implementing the selectkbest algorithm with 2000 and 10000 genes, then reducing those numbers using lasso
lasso_gene_list = []
X_train_lasso_list = []
X_test_lasso_list = []

for i in [2000, 10000]:
    pre_filter = SelectKBest(score_func=f_classif, k=i)
    X_train_pre = pre_filter.fit_transform(X_train_s, y_train)
    X_test_pre = pre_filter.transform(X_test_s)
    pre_filtered_columns = X_train.columns[pre_filter.get_support()]

    lasso = LogisticRegression(l1_ratio=1, solver='saga', C=0.1, max_iter=5000, random_state=42, class_weight='balanced')
    lasso_selector = SelectFromModel(lasso)
    lasso_selector.fit(X_train_pre, y_train)

    X_train_lasso = lasso_selector.transform(X_train_pre)
    X_test_lasso = lasso_selector.transform(X_test_pre)
    lasso_genes = pre_filtered_columns[lasso_selector.get_support()]

    lasso_gene_list.append(lasso_genes)
    X_train_lasso_list.append(X_train_lasso)
    X_test_lasso_list.append(X_test_lasso)
    print(f"k={i}: {len(lasso_genes)} genes selected")



#Implementing the selectkbest algorithm with 2000 and 10000 genes, then reducing those numbers using elastic net
en_gene_list = []
X_train_en_list = []
X_test_en_list = []

for i in [2000, 10000]:
    pre_filter = SelectKBest(score_func=f_classif, k=i)
    X_train_pre = pre_filter.fit_transform(X_train_s, y_train)
    X_test_pre = pre_filter.transform(X_test_s)
    pre_filtered_columns = X_train.columns[pre_filter.get_support()]

    en = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        C=0.1,
        max_iter=5000,
        class_weight="balanced",
        random_state=42,
        verbose=1
    )
    en_selector = SelectFromModel(en)
    en_selector.fit(X_train_pre, y_train)

    X_train_en = en_selector.transform(X_train_pre)
    X_test_en = en_selector.transform(X_test_pre)
    en_genes = pre_filtered_columns[en_selector.get_support()]

    en_gene_list.append(en_genes)
    X_train_en_list.append(X_train_en)
    X_test_en_list.append(X_test_en)
    print(f"k={i}: {len(en_genes)} genes selected")








#Main function, implementing and displaying the results of each ML model, for both k=2000 and k=10000, and lasso and elastic net
def model_pipline(model_name, model_cls, model_kwargs, X_train_list, X_test_list, y_train, y_test, gene_list_name, gene_list):
    for idx, k in enumerate([2000, 10000]):
        model = model_cls(**model_kwargs)
        model.fit(X_train_list[idx], y_train)
        preds = model.predict(X_test_list[idx])

        print(f'{model_name} ({gene_list_name}, {len(gene_list[idx])} genes) test accuracy: {accuracy_score(y_test, preds):.3f}')
        print(classification_report(y_test, preds))
        ConfusionMatrixDisplay.from_estimator(model, X_test_list[idx], y_test)
        plt.title(f'{model_name} Confusion Matrix ({gene_list_name}, {len(gene_list[idx])} genes)')
        plt.show()

#%%

#running each model for lassso and 

model_pipline(
    model_name='Decision Tree',
    model_cls = DecisionTreeClassifier,
    model_kwargs = {'random_state': 42},
    X_train_list = X_train_lasso_list,
    X_test_list=X_test_lasso_list,
    y_train = y_train,
    y_test = y_test,
    gene_list_name='lasso',
    gene_list=lasso_gene_list
)  


model_pipline(
    model_name='SVM',
    model_cls=SVC,
    model_kwargs={'random_state': 42, 'class_weight': 'balanced'},
    X_train_list = X_train_lasso_list,
    X_test_list=X_test_lasso_list,
    y_train = y_train,
    y_test = y_test,
    gene_list_name='lasso',
    gene_list=lasso_gene_list
)


model_pipline(
    model_name='ADA',
    model_cls=AdaBoostClassifier,
    model_kwargs={'random_state': 42},
    X_train_list = X_train_lasso_list,
    X_test_list=X_test_lasso_list,
    y_train = y_train,
    y_test = y_test,
    gene_list_name='lasso',
    gene_list=lasso_gene_list
)

model_pipline(
    model_name='KNN',
    model_cls=KNeighborsClassifier,
    model_kwargs={'n_neighbors':10},
    X_train_list = X_train_lasso_list,
    X_test_list=X_test_lasso_list,
    y_train = y_train,
    y_test = y_test,
    gene_list_name='lasso',
    gene_list=lasso_gene_list
)


model_pipline(
    model_name='Decision Tree',
    model_cls = DecisionTreeClassifier,
    model_kwargs = {'random_state': 42},
    X_train_list = X_train_en_list,
    X_test_list=X_test_en_list,
    y_train = y_train,
    y_test = y_test,
    gene_list_name='en',
    gene_list=en_gene_list
) 


model_pipline(
    model_name='SVM',
    model_cls=SVC,
    model_kwargs={'random_state': 42, 'class_weight': 'balanced'},
    X_train_list = X_train_en_list,
    X_test_list=X_test_en_list,
    y_train = y_train,
    y_test = y_test,
    gene_list_name='en',
    gene_list=en_gene_list
)


model_pipline(
    model_name='ADA',
    model_cls=AdaBoostClassifier,
    model_kwargs={'random_state': 42},
    X_train_list = X_train_en_list,
    X_test_list=X_test_en_list,
    y_train = y_train,
    y_test = y_test,
    gene_list_name='en',
    gene_list=en_gene_list
)

model_pipline(
    model_name='KNN',
    model_cls=KNeighborsClassifier,
    model_kwargs={'n_neighbors':10},
    X_train_list = X_train_en_list,
    X_test_list=X_test_en_list,
    y_train = y_train,
    y_test = y_test,
    gene_list_name='en',
    gene_list=en_gene_list
)


