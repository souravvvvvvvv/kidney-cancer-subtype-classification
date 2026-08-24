This project uses machine learning models to classify the kidney cancer subtypes: KIRC, KICH, and KIRP

The datasets used for this project can be found here:
This project loosely follows the pipline found here, with many notable alterations
The full methodology and results can be found here:

In summary, pre-filtering techniques such as lasso regression and elastic net were used to filter the genes in the dataset to ones that are important for classifying kidney cancer subtypes
Then, those genes were passed to the ML models, with a 70/30 train test split, and the performance of those models were evaluated. 

