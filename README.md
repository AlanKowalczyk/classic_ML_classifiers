# Classic ML Classifiers

## Overview  
A rigorous benchmarking study evaluating 16 machine learning algorithms across 6 distinct feature extraction techniques for financial text classification.

## Approach  
- Prepared tabular datasets
- Trained multiple classifiers (e.g., Logistic Regression, SVM, kNN, Random Forest)
- Feature Engineering: * Sparse: Bag of Words (BoW), TF-IDF, N-grams (1-2), Char-grams (3-5).
                       * Dense: Word2Vec embeddings (TF-IDF weighted), LSA (TruncatedSVD).
- Evaluated using cross-validation with Optuna tuning

## Results  
- Model performance metrics (accuracy, F1)
- Visual confusion matrices

## Tools & Techniques  
Python, scikit-learn, pandas, matplotlib


## Learning Outcomes  
Shows ability to compare models and interpret performance metrics.


