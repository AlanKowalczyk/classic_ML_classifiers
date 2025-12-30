"""
Created on 03 Oct 2025
Last modification 28 Oct 2025 by AK

@author Alan Kowalczyk
"""

import os
import time
import multiprocessing
import re
import warnings
import spacy

import pandas as pd
import numpy as np
import matplotlib
import ast

# Use a non-interactive backend to avoid Tkinter cleanup warnings during batch plotting
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import nltk

from matplotlib.lines import Line2D

from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import (
    SGDClassifier,
    RidgeClassifier,
    Perceptron,
    LogisticRegression
)
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
)

from scipy import sparse
from gensim.models import Word2Vec, FastText
import gensim.downloader as gensim_api

import optuna
import json
from optuna.integration import LightGBMPruningCallback

SKOPT_AVAILABLE = True


current_directory = os.path.dirname(os.path.realpath(__file__))
num_cores = multiprocessing.cpu_count()  # Get the number of CPU cores on the current machine
warnings.filterwarnings("ignore")
today_date = dt.date.today()


CLASSIFIER_MARKERS = {
    "LightGBM": "o",
    "Logistic Regression": "s",
    "Naive Bayes": "^",
    "Support Vector Classifier": "v",
    "Random Forest Classifier": "<",
    "Decision Tree Classifier": ">",
    "k-NN Classifier": "P",
    "MLP Classifier": "X",
    "Bagging Classifier": "D",
    "Perceptron": "*",
    "SGD Classifier": "h",
    "Extra Trees Classifier": "H",
    "Ridge Classifier": "+",
    "AdaBoost Classifier": "x",
    "Gradient Boosting Classifier": "d",
    "XGBoost": "|",
}

FEATURE_SET_COLORS = {
    "BoW": "#1f77b4",
    "TF-IDF": "#ff7f0e",
    "TF-IDF 1-2gram": "#2ca02c",
    "Word2Vec": "#d62728",
    "FastText": "#9467bd",
    "GloVe": "#8c564b",
}

def clean_text(text): # last change 03 Oct
    """ 
    clean_text - function for pre-processing text for classic ML methods
    input:
        text - a string - text to be cleaned
    output:
             - a string - cleaned text
    """
    # removing URLs
    text = re.sub(r"http\S+|www\S+|https\S+",
                  "",
                  text,
                  flags = re.MULTILINE)

    # removing HTML tags
    text = re.sub(r"<.*?>", "", text)

    # removing numbers
    text = re.sub(r"\d+", "", text)
    
    # lowercasing the text
    text = text.lower()

    # removing punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)

    # removing extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # removing stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    filtered_text = " ".join(filtered_words)
    
    # lemmatisation
    doc = spacy_nlp(filtered_text)
    lemmatized_words = [token.lemma_ for token in doc]
    
    return " ".join(lemmatized_words)


def feature_extraction(df, vectorizer):# last change 03 Oct
    """
    feature_extraction - function which implements the feature extraction
    input:
         df - a dataframe - dataframe with clean text data
         vectorizer - an object - an object for changing text data into numerical values
    output:
        dfe - a dataframe  - a dataframe with added extracted features
        matrix - matrix - the matrix of features 
    """
    # fit and transform the text data to features
    matrix = vectorizer.fit_transform(df['Cleaned_text'])


def compute_document_embeddings(token_lists, embedding_lookup, vector_size, lowercase_fallback=False):
    """Average token embeddings per document to obtain fixed-length vectors."""
    doc_embeddings = np.zeros((len(token_lists), vector_size), dtype=np.float32)
    for idx, tokens in enumerate(token_lists):
        vectors = []
        for token in tokens:
            if not isinstance(token, str):
                token = str(token)
            if token in embedding_lookup:
                vectors.append(embedding_lookup[token])
            elif lowercase_fallback and token.lower() in embedding_lookup:
                vectors.append(embedding_lookup[token.lower()])
        if vectors:
            doc_embeddings[idx] = np.mean(vectors, axis=0)
    return doc_embeddings


def build_embedding_feature_sets(token_lists, vector_size=100):
    """Construct dense embedding feature matrices using Word2Vec, FastText, and GloVe."""
    embedding_features = {}

    print("Training Word2Vec embeddings...")
    w2v_model = Word2Vec(sentences=token_lists,
                         vector_size=vector_size,
                         window=5,
                         min_count=1,
                         workers=max(1, num_cores - 1))
    embedding_features["Word2Vec"] = compute_document_embeddings(token_lists, w2v_model.wv, vector_size)

    print("Training FastText embeddings...")
    fasttext_model = FastText(sentences=token_lists,
                              vector_size=vector_size,
                              window=5,
                              min_count=1,
                              workers=max(1, num_cores - 1))
    embedding_features["FastText"] = compute_document_embeddings(token_lists, fasttext_model.wv, vector_size)

    print("Loading GloVe embeddings...")
    try:
        glove_vectors = gensim_api.load(f"glove-wiki-gigaword-{vector_size}")
        embedding_features["GloVe"] = compute_document_embeddings(token_lists, glove_vectors, glove_vectors.vector_size, lowercase_fallback=True)
    except Exception as glove_error:
        print(f"Unable to load GloVe embeddings: {glove_error}")

    return embedding_features


def supervised_ml(feature_sets): # last change 13 Oct
    """
    supervised_ml - a function which creates a pipeline for ML models, and save results fo file, create confusion matrices
    input:
        feature_sets - a dictionary - dictionary containing matrices with features
    output:
        results_df - a dataframe - a dataframe containing results
    """
    warnings.filterwarnings("ignore")
    # dictionary of supervised ML classifiers
    classifiers = {
        "LightGBM": LGBMClassifier(verbosity = -1),
        "Logistic Regression": LogisticRegression(verbose = 0),
        "Naive Bayes": MultinomialNB(),
        "Support Vector Classifier": SVC(verbose = 0),
        "Random Forest Classifier": RandomForestClassifier(verbose = 0), 
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "k-NN Classifier": KNeighborsClassifier(),
        "MLP Classifier": MLPClassifier(verbose = 0),
        "Bagging Classifier": BaggingClassifier(),
        "Perceptron": Perceptron(verbose = 0),
        "SGD Classifier": SGDClassifier(verbose = 0),
        "Extra Trees Classifier": ExtraTreesClassifier(),
        "Ridge Classifier": RidgeClassifier(),
        "AdaBoost Classifier": AdaBoostClassifier(),
        "Gradient Boosting Classifier": GradientBoostingClassifier(verbose = 0),    
        "XGBoost": XGBClassifier(verbosity = 0)
    }
    output_dir = f"predictions_{today_date}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []
    optuna_timings = []

    # loop through each feature extraction method
    for feature_name, X in feature_sets.items():
        feature_hpo_time = 0.0
        is_sparse_matrix = sparse.issparse(X)

        # split the data into training and testing sets (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

        # scale the features using StandardScaler (sparse-friendly) or MinMaxScaler for dense matrices
        if is_sparse_matrix:
            scaler = StandardScaler(with_mean=False)
        else:
            scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_train = y_train.reset_index(drop=True)

        print("scikit-optimize detected: running Bayesian optimization for each classifier")
        best_classifiers = {}
        for class_name, classifier in classifiers.items():
            tuned_classifier, hpo_time = calculate_params_bayesian(classifier, class_name, X_train, y_train, feature_name)
            best_classifiers[class_name] = tuned_classifier
            feature_hpo_time += hpo_time

        print_time_taken(feature_hpo_time, f"Optuna tuning for {feature_name}")

        print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Train and evaluate each classifier with features set in parallel computation
        parallel_results = Parallel(n_jobs=num_cores)(
            delayed(classifier_model)(classifier, class_name, X_train, X_test, y_train, y_test)
            for class_name, classifier in best_classifiers.items()
        )

        for result, (class_name, classifier) in zip(parallel_results, best_classifiers.items()):
            result['Feature Set'] = feature_name
            results.append(result)

            # saving prediction to files for each classifier
            predictions_df = pd.DataFrame({
                "True Labels": y_test,
                "Predicted Labels": result['Predictions']
            })
            file_name_pred = f"{class_name}_{feature_name}_predictions.csv"
            file_path_pred = os.path.join(output_dir, file_name_pred)
            predictions_df.to_csv(file_path_pred, index=False)

            # saving confusion matrices to files
            plot_confusion_matrix(result['Confusion Matrix'], class_name, feature_name)

        optuna_timings.append({
            "Feature Set": feature_name,
            "Optuna Time [s]": feature_hpo_time
        })

    results_df = pd.DataFrame(results)

    timing_file = f"optuna_feature_timings_{today_date}.txt"
    with open(timing_file, 'w') as timing_fh:
        timing_fh.write("Optuna runtime per feature set\n")
        for entry in optuna_timings:
            timing_fh.write(f"{entry['Feature Set']}: {entry['Optuna Time [s]']:.2f} seconds\n")
    print(f"Optuna runtime summary written to {timing_file}")

    return results_df

def print_time_taken(seconds, task):# last change 17 oct
    """
    print_time_taken - a function which is responsible for printing information in format hh:mm:ss.ss
    input:
        seconds - a float - the number of seconds required to finish the task
        task - a string - the name of the task
    output:
        none

    """
    if seconds < 60:
        formatted_time = f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        formatted_time = f"{minutes} minutes and {seconds:.2f} seconds"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        formatted_time = f"{hours} hours, {minutes} minutes, and {seconds:.2f} seconds"
        
    print(f"Time taken for {task}: {formatted_time}")
    

def calculate_params_bayesian(classifier, class_name, X_train, y_train, feature_name, n_trials=2, timeout=None, optimize_metric='f1_weighted'):
    """
    Optuna-based hyperparameter optimisation for publication2.py.

    - Uses Optuna TPE sampler with MedianPruner to efficiently search hyperparameters.
    - Optimizes weighted F1 by default using StratifiedKFold CV.
    - Supports the 16 classifiers defined in `supervised_ml`.

    Parameters:
      classifier - prototype classifier instance (unused)
      class_name - string name of classifier
    X_train, y_train - training data
    feature_name - name of feature representation (e.g., BoW, TF-IDF)
      n_trials - number of Optuna trials (default 200, increase for better results)
      timeout - optional timeout in seconds for the study
      optimize_metric - 'f1_weighted' or 'accuracy'

    Returns: tuple(classifier, elapsed_time_seconds)
    """

    warnings.filterwarnings("ignore")
    start_time = time.time()

    summary_dir = os.path.join("hpo_summaries", feature_name)
    summary_file = os.path.join(summary_dir, f"hpo_summary_{feature_name}_{today_date}.csv")

    def build_final_classifier(best_params):
        if class_name == "LightGBM":
            return LGBMClassifier(**best_params, random_state=42, verbosity=-1)
        if class_name == "Logistic Regression":
            return LogisticRegression(**best_params, max_iter=2000, random_state=42)
        if class_name == "Naive Bayes":
            return MultinomialNB(**best_params)
        if class_name == "Support Vector Classifier":
            return SVC(**best_params, probability=False, random_state=42)
        if class_name == "Random Forest Classifier":
            return RandomForestClassifier(**best_params, random_state=42, n_jobs=1)
        if class_name == "Decision Tree Classifier":
            return DecisionTreeClassifier(**best_params, random_state=42)
        if class_name == "k-NN Classifier":
            return KNeighborsClassifier(**best_params)
        if class_name == "MLP Classifier":
            n_hidden = best_params.get('n_hidden', 1)
            h1 = best_params.get('h1')
            h2 = best_params.get('h2')
            if n_hidden == 1:
                hidden = (h1,)
            elif n_hidden == 2:
                hidden = (h1, h2)
            else:
                hidden = (h1, h2, int(h2/2))
            mlp_params = best_params.copy()
            for k in ['h1', 'h2', 'n_hidden']:
                mlp_params.pop(k, None)
            mlp_params['hidden_layer_sizes'] = hidden
            if 'max_iter' not in mlp_params:
                mlp_params['max_iter'] = 2000
            return MLPClassifier(**mlp_params, random_state=42)
        if class_name == "Bagging Classifier":
            return BaggingClassifier(**best_params, random_state=42)
        if class_name == "Perceptron":
            return Perceptron(**best_params, random_state=42)
        if class_name == "SGD Classifier":
            return SGDClassifier(**best_params, random_state=42)
        if class_name == "Extra Trees Classifier":
            return ExtraTreesClassifier(**best_params, random_state=42)
        if class_name == "Ridge Classifier":
            return RidgeClassifier(**best_params)
        if class_name == "AdaBoost Classifier":
            return AdaBoostClassifier(**best_params, random_state=42)
        if class_name == "Gradient Boosting Classifier":
            return GradientBoostingClassifier(**best_params, random_state=42)
        if class_name == "XGBoost":
            return XGBClassifier(**best_params, random_state=42, verbosity=0)
        raise ValueError(f"Unsupported classifier after Optuna tuning: {class_name}")

    if os.path.exists(summary_file):
        try:
            cached_df = pd.read_csv(summary_file)
            cached_df['BestScore'] = pd.to_numeric(cached_df['BestScore'], errors='coerce')
            cached_rows = cached_df[cached_df['Classifier'] == class_name]
            if not cached_rows.empty:
                cached_row = cached_rows.sort_values('BestScore', ascending=False).iloc[0]
                params_str = str(cached_row['BestParams']).strip()
                if params_str.startswith("'") and params_str.endswith("'"):
                    params_str = params_str[1:-1]
                try:
                    best_params = json.loads(params_str)
                except json.JSONDecodeError:
                    best_params = ast.literal_eval(params_str)
                print(f"Loaded cached hyperparameters for {class_name} ({feature_name}) from {summary_file}")
                return build_final_classifier(best_params), 0.0
            else:
                print(f"No cached hyperparameters for {class_name} ({feature_name}) found in {summary_file}; running Optuna.")
        except Exception as cache_error:
            print(f"Failed to load cached hyperparameters for {class_name} ({feature_name}) due to {cache_error}; running Optuna.")

    # Ensure X is indexable
    try:
        Xc = X_train.tocsc().copy()
    except Exception:
        Xc = X_train

    def objective(trial):
        try:
            # Suggest params per classifier
            if class_name == "LightGBM":
                # Use conservative ranges to keep trials fast and avoid degenerate trees
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.1, log=True)
                num_leaves = trial.suggest_int('num_leaves', 8, 128)
                max_depth = trial.suggest_int('max_depth', -1, 16)
                subsample = trial.suggest_float('subsample', 0.6, 1.0)
                min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 50)
                reg_alpha = trial.suggest_float('reg_alpha', 1e-6, 1.0, log=True)
                reg_lambda = trial.suggest_float('reg_lambda', 1e-6, 1.0, log=True)
                model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                                       max_depth=max_depth, subsample=subsample, min_data_in_leaf=min_data_in_leaf,
                                       reg_alpha=reg_alpha, reg_lambda=reg_lambda, random_state=42, n_jobs=1, verbosity=-1)

            elif class_name == "Logistic Regression":
                C = trial.suggest_float('C', 1e-4, 1e2, log=True)
                solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])
                model = LogisticRegression(C=C, solver=solver, max_iter=2000, random_state=42)

            elif class_name == "Naive Bayes":
                alpha = trial.suggest_float('alpha', 1e-3, 1.0, log=True)
                model = MultinomialNB(alpha=alpha)

            elif class_name == "Support Vector Classifier":
                C = trial.suggest_float('C', 1e-4, 1e2, log=True)
                kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
                model = SVC(C=C, kernel=kernel, probability=False, random_state=42)

            elif class_name == "Random Forest Classifier":
                n_estimators = trial.suggest_int('n_estimators', 50, 1000)
                max_depth = trial.suggest_int('max_depth', 3, 100)
                max_features = trial.suggest_float('max_features', 0.1, 1.0)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=42, n_jobs=1)

            elif class_name == "Decision Tree Classifier":
                max_depth = trial.suggest_int('max_depth', 1, 100)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)

            elif class_name == "k-NN Classifier":
                n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
                p = trial.suggest_categorical('p', [1, 2])
                model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p)

            elif class_name == "MLP Classifier":
                h1 = trial.suggest_int('h1', 50, 512)
                n_hidden = trial.suggest_categorical('n_hidden', [1,2,3])
                h2 = trial.suggest_int('h2', 50, 512)
                alpha = trial.suggest_float('alpha', 1e-7, 1e-1, log=True)
                lr = trial.suggest_float('learning_rate_init', 1e-5, 1.0, log=True)
                if n_hidden == 1:
                    hidden = (h1,)
                elif n_hidden == 2:
                    hidden = (h1, h2)
                else:
                    hidden = (h1, h2, int(h2/2))
                model = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=lr, max_iter=2000, random_state=42)

            elif class_name == "Bagging Classifier":
                n_estimators = trial.suggest_int('n_estimators', 10, 200)
                model = BaggingClassifier(n_estimators=n_estimators, random_state=42)

            elif class_name == "Perceptron":
                penalty = trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet', None])
                alpha = trial.suggest_float('alpha', 1e-6, 1e-2, log=True)
                fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
                max_iter = trial.suggest_int('max_iter', 200, 2000)
                tol = trial.suggest_float('tol', 1e-5, 1e-2, log=True)
                shuffle = trial.suggest_categorical('shuffle', [True, False])
                early_stopping = trial.suggest_categorical('early_stopping', [True, False])
                validation_fraction = trial.suggest_float('validation_fraction', 0.05, 0.2) if early_stopping else 0.1
                n_iter_no_change = trial.suggest_int('n_iter_no_change', 5, 20) if early_stopping else 10
                eta0 = trial.suggest_float('eta0', 1e-3, 1.0, log=True)
                l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0) if penalty == 'elasticnet' else 0.15
                model = Perceptron(penalty=penalty,
                                   alpha=alpha,
                                   fit_intercept=fit_intercept,
                                   max_iter=max_iter,
                                   tol=tol,
                                   shuffle=shuffle,
                                   early_stopping=early_stopping,
                                   validation_fraction=validation_fraction,
                                   n_iter_no_change=n_iter_no_change,
                                   eta0=eta0,
                                   l1_ratio=l1_ratio,
                                   random_state=42)

            elif class_name == "SGD Classifier":
                loss = trial.suggest_categorical('loss', ['hinge', 'log_loss', 'modified_huber'])
                alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
                model = SGDClassifier(loss=loss, alpha=alpha, max_iter=5000, random_state=42)

            elif class_name == "Extra Trees Classifier":
                n_estimators = trial.suggest_int('n_estimators', 50, 800)
                max_depth = trial.suggest_int('max_depth', 3, 100)
                model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

            elif class_name == "Ridge Classifier":
                alpha = trial.suggest_float('alpha', 1e-4, 10.0, log=True)
                model = RidgeClassifier(alpha=alpha)

            elif class_name == "AdaBoost Classifier":
                n_estimators = trial.suggest_int('n_estimators', 50, 500)
                learning_rate = trial.suggest_float('learning_rate', 1e-3, 1.0)
                model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

            elif class_name == "Gradient Boosting Classifier":
                n_estimators = trial.suggest_int('n_estimators', 50, 800)
                learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.5)
                max_depth = trial.suggest_int('max_depth', 2, 20)
                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)

            elif class_name == "XGBoost":
                n_estimators = trial.suggest_int('n_estimators', 50, 800)
                learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.5)
                max_depth = trial.suggest_int('max_depth', 2, 20)
                model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, verbosity=0)

            else:
                raise ValueError(f"Unsupported classifier in Optuna tuner: {class_name}")

            # Stratified CV
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in skf.split(Xc, y_train):
                X_tr = Xc[train_idx]
                y_tr = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
                X_val = Xc[val_idx]
                y_val = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]

                # Some models require dense input
                dense_required = class_name in ['Naive Bayes', 'k-NN Classifier', 'Perceptron', 'Ridge Classifier', 'Logistic Regression']
                try:
                    if dense_required and hasattr(X_tr, 'toarray'):
                        X_tr_use = X_tr.toarray()
                        X_val_use = X_val.toarray()
                    else:
                        X_tr_use = X_tr
                        X_val_use = X_val
                except Exception:
                    X_tr_use = X_tr
                    X_val_use = X_val

                # For LightGBM use early stopping + Optuna pruning to avoid very long fits
                if class_name == "LightGBM":
                    # Try the modern sklearn-fit signature with early stopping and pruning callback.
                    # Some LightGBM builds (or older versions) may not accept these kwargs; fall back if needed.
                    try:
                        model.fit(
                            X_tr_use,
                            y_tr,
                            eval_set=[(X_val_use, y_val)],
                            eval_metric='multi_logloss',
                            early_stopping_rounds=30,
                            callbacks=[LightGBMPruningCallback(trial, 'multi_logloss')],
                            verbose=False,
                        )
                    except TypeError as te:
                        # Fallback for versions that don't accept early_stopping_rounds/callbacks
                        if 'unexpected keyword argument' in str(te) or 'got an unexpected keyword argument' in str(te):
                            print(f"LightGBM.fit does not accept early_stopping_rounds/callbacks in this environment; falling back to simple fit. Error: {te}")
                            # Use a reduced n_estimators for safety to keep trial time reasonable
                            try:
                                model.set_params(n_estimators=min(getattr(model, 'n_estimators', 100), 200))
                            except Exception:
                                pass
                            model.fit(X_tr_use, y_tr)
                        else:
                            # re-raise other TypeErrors
                            raise
                    except Exception as e:
                        # If LightGBM raises for other reasons, surface the error to the trial
                        print(f"LightGBM fit error in trial: {e}")
                        raise
                else:
                    model.fit(X_tr_use, y_tr)
                preds = model.predict(X_val_use)
                if optimize_metric == 'f1_weighted':
                    score = f1_score(y_val, preds, average='weighted')
                else:
                    score = accuracy_score(y_val, preds)
                scores.append(score)

            mean_score = float(np.mean(scores))
            return mean_score

        except Exception as e:
            print(f"Optuna trial error for {class_name}: {e}")
            return 0.0

    # Create study
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

    print(f"Starting Optuna study for {class_name} with {n_trials} trials...")
    # Use in-process parallelism: pick a conservative number of workers to avoid oversubscription.
    try:
        num_workers = max(1, min(num_cores // 2, num_cores))
    except Exception:
        num_workers = 1
    print(f"Optuna will run with n_jobs={num_workers} (in-process parallel trials)")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=num_workers)

    best_params = study.best_params

    # Save HPO summary to CSV (append)
    os.makedirs(summary_dir, exist_ok=True)
    header = not os.path.exists(summary_file)
    with open(summary_file, 'a') as fh:
        if header:
            fh.write('FeatureSet,Classifier,BestScore,BestParams\n')
        fh.write(f"{feature_name},{class_name},{study.best_value},'{json.dumps(best_params)}'\n")

    final_clf = build_final_classifier(best_params)

    elapsed = time.time() - start_time
    print(f"Optuna HPO for {class_name} completed in {elapsed:.1f}s; best {optimize_metric}={study.best_value:.4f}")
    print(f"Best params: {best_params}")

    return final_clf, elapsed


def classifier_model(classifier, class_name, X_train, X_test, y_train, y_test): # last change 13 Oct
    """ 
    classifier_model - a function that trains all classic ML models and evaluates them
    input:
        classifier - an object - an actual classifier model to be trained
        class_name - a string - a classifier name
        X_train, y_train, X_test, y_test - dataframes - splited and scalled data for training and testing
    Output:
        metadata - a dictionary - predictions made by the models and evaluation of those predictions
    """
    warnings.filterwarnings("ignore")
    start_time = time.time()
    
    # train model
    classifier.fit(X_train, y_train)
    
    # make predictions
    X_test_copy = X_test.copy()
    predictions = classifier.predict(X_test_copy)
    end_time = time.time() - start_time
    
    # evaluate model
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average = "weighted")
    recall = recall_score(y_test, predictions, average = "weighted")
    f1 = f1_score(y_test, predictions, average = "weighted")
    conf_matrix = confusion_matrix(y_test, predictions)
    
    true_labels = y_test.tolist() if isinstance(y_test, pd.Series) else list(y_test)
     
    metadata = {"Model name": class_name,
                "Model": classifier,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 score": f1,
                "Confusion Matrix": conf_matrix,
                "Time [s]": end_time,
                "Predictions": predictions,
                "Label": true_labels}
    
    return metadata
   
   
def plot_confusion_matrix(conf_matrix, class_name, feature_name):#last change 13 Oct
    """
    plot_confusion_matrix - a function which is plotting confusion matrix in graphical form, and adding them to a latex file
    input:
        conf_matrix - a ndarray - numpy multidimensional array containing confusion matrix
        class_name - a string - a classifier name
        feature_name - a string - name of the feature
    output:
        none
    """

    output_dir = f"confusion_matrices_{today_date}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )

    plt.title(f"Confusion Matrix - {class_name} - {feature_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    output_file = os.path.join(output_dir, f"{class_name}_{feature_name}_confusion_matrix.png")
    plt.savefig(output_file)
    plt.close()

    tex_file = f"confusion_matrices_{today_date}.tex"
    with open(tex_file, 'a') as tex:
        tex.write(r"\begin{figure}[ht!]" + "\n")
        tex.write(f"    \\centering" + "\n")
        tex.write(f"    \\includegraphics[width=0.8\\textwidth]{{{output_file}}}" + "\n")
        tex.write(f"    \\caption{{Confusion Matrix - {class_name} - {feature_name}}}" + "\n")
        tex.write(r"\end{figure}" + "\n")


def plot_performance_scatter(results_df):
    """Generate a scatter plot comparing training time against F1 score across feature sets."""
    output_dir = f"performance_plots_{today_date}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(figsize=(10, 6))

    for feature_name, feature_slice in results_df.groupby('Feature Set'):
        feature_color = FEATURE_SET_COLORS.get(feature_name, '#333333')
        for model_name, model_slice in feature_slice.groupby('Model name'):
            marker = CLASSIFIER_MARKERS.get(model_name, 'o')
            ax.scatter(
                model_slice['Time [s]'],
                model_slice['F1 score'],
                color=feature_color,
                marker=marker,
                s=70,
                edgecolors='black',
                linewidths=0.5,
                alpha=0.9
            )

    ax.set_title('Model Performance vs Training Time')
    ax.set_xlabel('Training Time [s]')
    ax.set_ylabel('F1 score')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

    classifier_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle='None',
            markerfacecolor='black',
            markeredgecolor='black',
            markersize=7,
            label=name
        )
        for name, marker in CLASSIFIER_MARKERS.items()
    ]

    feature_handles = [
        Line2D(
            [0],
            [0],
            marker='o',
            linestyle='None',
            markerfacecolor=color,
            markeredgecolor='black',
            markersize=8,
            label=name
        )
        for name, color in FEATURE_SET_COLORS.items()
    ]

    marker_legend = ax.legend(
        handles=classifier_handles,
        title='Classifiers (symbols)',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        frameon=False
    )

    color_legend = ax.legend(
        handles=feature_handles,
        title='Feature Sets (colours)',
        bbox_to_anchor=(1.02, 0.45),
        loc='upper left',
        frameon=False
    )

    ax.add_artist(marker_legend)
    fig.tight_layout()
    scatter_path = os.path.join(output_dir, f"performance_scatter_{today_date}.png")
    fig.savefig(scatter_path)
    plt.close(fig)

# licence for dataset Attribution-NonCommercial-ShareAlike 4.0 International
# credit to:
# Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: 
# Detecting semantic orientations in economic texts. Journal of the Association for Information 
# Science and Technology, 65(4), 782-796.

# Read Sentences_AllAgree.csv robustly: handle both header and headerless formats
dfl = pd.read_csv("Sentences_AllAgree.csv",
                  header = None,
                  names = ['Label', 'Text'],
                  encoding = 'ISO-8859-1') # dfl - data frame with labelled data

dfl = dfl.dropna().reset_index(drop=True)  # Drop rows with NaN values and reset index

#print(dfl.info())
#print(dfl.head())

# Download required NLTK data for stopwords if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

stop_words = set(stopwords.words("english"))
spacy_nlp = spacy.load("en_core_web_sm")

# Clean text for ML algorithms
dfl['Cleaned_text'] = dfl['Text'].apply(lambda x: clean_text(x))

# initialize vectorizers
bow_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()
tfidf_bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)

df_bow, bow_matrix = feature_extraction(dfl, bow_vectorizer)
df_tfidf, tfidf_matrix = feature_extraction(dfl, tfidf_vectorizer)
df_tfidf_bigram, tfidf_bigram_matrix = feature_extraction(dfl, tfidf_bigram_vectorizer)

# Tokenise raw text for embedding-based features (non-cleaned)
token_lists_raw = dfl['Text'].astype(str).apply(word_tokenize).tolist()
embedding_feature_sets = build_embedding_feature_sets(token_lists_raw)

# Features and labels
feature_sets = {
    "BoW": bow_matrix,
    "TF-IDF": tfidf_matrix,
    "TF-IDF 1-2gram": tfidf_bigram_matrix
}
for name, matrix in embedding_feature_sets.items():
    feature_sets[name] = matrix
y = dfl['Label']
# Convert labels as XGBoost have requirements of [0,1,2] not [-1,0,1]
y = dfl['Label'].map({'negative': 0, 'neutral': 1, 'positive': 2})


print(dfl.head())
print(y.value_counts())
print(bow_matrix.shape)
print(tfidf_matrix.shape)
print(tfidf_bigram_matrix.shape)
for name, matrix in embedding_feature_sets.items():
    print(f"{name} {np.array(matrix).shape}")


results_df = supervised_ml(feature_sets)

cols = ['Feature Set', 'Model name', 'F1 score', 'Time [s]']
results_df['Time [s]'] = results_df['Time [s]'].round(2)
sorted_df = results_df.sort_values(by = ['Feature Set', 'F1 score'], ascending = [True, False])
print(sorted_df[cols]) 

results_df.to_csv(f"ml_results_{today_date}.csv", index=False)

# Generate aggregate scatter plot to visualise trade-offs between runtime and performance
plot_performance_scatter(sorted_df[['Feature Set', 'Model name', 'F1 score', 'Time [s]']].copy())
