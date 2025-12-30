# project2.py
"""
Project2 - Classic ML comparison
Created on 03 Oct 2025 (adapted from publication2.py)
Last modification: 28 Oct 2025
Author: Alan Kowalczyk (adaptation)
"""

import os
import time
import multiprocessing
import re
import warnings
import spacy
import ast
import json

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import nltk

from matplotlib.lines import Line2D
from joblib import Parallel, delayed

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier, Perceptron, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier

from scipy import sparse
from gensim.models import Word2Vec, FastText
import gensim.downloader as gensim_api

import optuna
from optuna.integration import LightGBMPruningCallback

# --------------------
# Config & globals
# --------------------
current_directory = os.path.dirname(os.path.realpath(__file__))
num_cores = multiprocessing.cpu_count()
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
    "Char 3-5gram": "#17becf",
    "TF-IDF + LSA": "#9467bd",
    "TF-IDF-weighted-Emb": "#8c564b",
}

# --------------------
# Utility functions
# --------------------
def clean_text(text):
    """Clean & lemmatize text for classic ML pipelines."""
    if not isinstance(text, str):
        text = str(text)
    # remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    # remove numbers
    text = re.sub(r"\d+", "", text)
    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    # remove stopwords
    words = text.split()
    filtered_words = [w for w in words if w not in stop_words]
    filtered_text = " ".join(filtered_words)
    # lemmatize
    doc = spacy_nlp(filtered_text)
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)

def compute_document_embeddings(token_lists, embedding_lookup, vector_size, lowercase_fallback=False):
    """Average token embeddings per document (simple mean)."""
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

def compute_tfidf_weighted_embeddings(token_lists, tfidf_vectorizer, embedding_lookup, vector_size, lowercase_fallback=True):
    """
    Compute TF-IDF-weighted averaged embeddings for each document.
    - token_lists: list[list[str]] tokens per doc (should match tfidf tokenization)
    - tfidf_vectorizer: fitted TfidfVectorizer
    - embedding_lookup: gensim KeyedVectors-like (embedding_lookup[word] -> vector)
    """
    # Prepare idf mapping for vocabulary tokens
    vocab = tfidf_vectorizer.get_feature_names_out()
    idf = dict(zip(vocab, tfidf_vectorizer.idf_))

    doc_embeddings = np.zeros((len(token_lists), vector_size), dtype=np.float32)
    for i, tokens in enumerate(token_lists):
        weighted_vecs = []
        weights = []
        for tok in tokens:
            key = tok
            if key in embedding_lookup and key in idf:
                w = idf[key]
                weighted_vecs.append(embedding_lookup[key] * w)
                weights.append(w)
            elif lowercase_fallback and tok.lower() in embedding_lookup and tok.lower() in idf:
                w = idf[tok.lower()]
                weighted_vecs.append(embedding_lookup[tok.lower()] * w)
                weights.append(w)
        if weighted_vecs and np.sum(weights) > 0:
            doc_embeddings[i] = np.sum(weighted_vecs, axis=0) / (np.sum(weights) + 1e-9)
        else:
            doc_embeddings[i] = np.zeros(vector_size, dtype=np.float32)
    return doc_embeddings

def feature_extraction(df, vectorizer):
    """Fit-transform cleaned text to feature matrix; return dense/sparse matrix."""
    matrix = vectorizer.fit_transform(df['Cleaned_text'])
    return matrix

# --------------------
# ML pipeline functions (adapted from your publication2.py)
# --------------------
def classifier_model(classifier, class_name, X_train, X_test, y_train, y_test):
    """Train classifier and evaluate (returns metadata dictionary)."""
    warnings.filterwarnings("ignore")
    start_time = time.time()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    end_time = time.time() - start_time

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(y_test, predictions)

    true_labels = y_test.tolist() if isinstance(y_test, pd.Series) else list(y_test)

    metadata = {
        "Model name": class_name,
        "Model": classifier,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 score": f1,
        "Confusion Matrix": conf_matrix,
        "Time [s]": end_time,
        "Predictions": predictions,
        "Label": true_labels
    }
    return metadata

def plot_confusion_matrix(conf_matrix, class_name, feature_name):
    """Save confusion matrix figure + append LaTeX snippet (same as original)."""
    output_dir = f"confusion_matrices_{today_date}"
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
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
        tex.write("    \\centering" + "\n")
        tex.write(f"    \\includegraphics[width=0.8\\textwidth]{{{output_file}}}" + "\n")
        tex.write(f"    \\caption{{Confusion Matrix - {class_name} - {feature_name}}}" + "\n")
        tex.write(r"\end{figure}" + "\n")

def print_time_taken(seconds, task):
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

# --- Optuna HPO function (kept with minimal changes) ---
def calculate_params_bayesian(classifier, class_name, X_train, y_train, feature_name, n_trials=2, timeout=None, optimize_metric='f1_weighted'):
    """
    Optuna-based hyperparameter optimisation.
    Kept similar to your original calculate_params_bayesian for compatibility.
    """
    warnings.filterwarnings("ignore")
    start_time = time.time()

    # summary path to cache best params per feature/classifier
    summary_dir = os.path.join("hpo_summaries", feature_name)
    os.makedirs(summary_dir, exist_ok=True)
    summary_file = os.path.join(summary_dir, f"hpo_summary_{feature_name}_{today_date}.csv")

    def build_final_classifier(best_params):
        # replicate mapping like your previous script
        if class_name == "LightGBM":
            return LGBMClassifier(**best_params, random_state=42, verbosity=-1)
        if class_name == "Logistic Regression":
            return LogisticRegression(**best_params, max_iter=500, random_state=42)
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
            # handle hidden layers params
            n_hidden = best_params.get('n_hidden', 1)
            h1 = best_params.get('h1', 100)
            h2 = best_params.get('h2', 50)
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

    # try to load cached best params
    if os.path.exists(summary_file):
        try:
            cached_df = pd.read_csv(summary_file)
            cached_df['BestScore'] = pd.to_numeric(cached_df['BestScore'], errors='coerce')
            cached_rows = cached_df[cached_df['Classifier'] == class_name]
            if not cached_rows.empty:
                cached_row = cached_rows.sort_values('BestScore', ascending=False).iloc[0]
                params_str = str(cached_row['BestParams']).strip()
                try:
                    best_params = json.loads(params_str) if params_str.startswith('{') else ast.literal_eval(params_str)
                    print(f"Loaded cached hyperparameters for {class_name} ({feature_name}) from {summary_file}")
                    return build_final_classifier(best_params), 0.0
                except Exception:
                    pass
        except Exception:
            pass

    # ensure indexable X for CV
    try:
        Xc = X_train.tocsc().copy()
    except Exception:
        Xc = X_train

    def objective(trial):
        try:
            # suggestions per classifier (kept conservative ranges)
            if class_name == "LightGBM":
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.1, log=True)
                num_leaves = trial.suggest_int('num_leaves', 8, 128)
                model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                                       random_state=42, n_jobs=1, verbosity=-1)
            elif class_name == "Logistic Regression":
                C = trial.suggest_float('C', 1e-2, 1e2, log=True)
                solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])
                model = LogisticRegression(C=C, solver=solver, max_iter=500, random_state=42)
            elif class_name == "Naive Bayes":
                alpha = trial.suggest_float('alpha', 1e-3, 1.0, log=True)
                model = MultinomialNB(alpha=alpha)
            elif class_name == "Support Vector Classifier":
                C = trial.suggest_float('C', 1e-4, 1e2, log=True)
                kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
                model = SVC(C=C, kernel=kernel, probability=False, random_state=42)
            elif class_name == "Random Forest Classifier":
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                max_depth = trial.suggest_int('max_depth', 3, 50)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=1)
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
                n_hidden = trial.suggest_categorical('n_hidden', [1, 2, 3])
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
                model = Perceptron(penalty=penalty, alpha=alpha, random_state=42)
            elif class_name == "SGD Classifier":
                loss = trial.suggest_categorical('loss', ['hinge', 'log_loss', 'modified_huber'])
                alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
                model = SGDClassifier(loss=loss, alpha=alpha, max_iter=5000, random_state=42)
            elif class_name == "Extra Trees Classifier":
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                max_depth = trial.suggest_int('max_depth', 3, 50)
                model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif class_name == "Ridge Classifier":
                alpha = trial.suggest_float('alpha', 1e-4, 10.0, log=True)
                model = RidgeClassifier(alpha=alpha)
            elif class_name == "AdaBoost Classifier":
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                learning_rate = trial.suggest_float('learning_rate', 1e-3, 1.0)
                model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
            elif class_name == "Gradient Boosting Classifier":
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.5)
                max_depth = trial.suggest_int('max_depth', 2, 20)
                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
            elif class_name == "XGBoost":
                n_estimators = trial.suggest_int('n_estimators', 50, 300)
                learning_rate = trial.suggest_float('learning_rate', 1e-3, 0.5)
                max_depth = trial.suggest_int('max_depth', 2, 20)
                model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, verbosity=0)
            else:
                raise ValueError(f"Unsupported classifier in Optuna tuner: {class_name}")

            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in skf.split(Xc, y_train):
                X_tr = Xc[train_idx]
                X_val = Xc[val_idx]
                y_tr = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
                y_val = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]

                dense_required = class_name in ['Naive Bayes', 'k-NN Classifier', 'Perceptron', 'Ridge Classifier', 'Logistic Regression']
                if dense_required and hasattr(X_tr, 'toarray'):
                    X_tr_use = X_tr.toarray()
                    X_val_use = X_val.toarray()
                else:
                    X_tr_use = X_tr
                    X_val_use = X_val

                if class_name == "LightGBM":
                    try:
                        model.fit(
                            X_tr_use, y_tr,
                            eval_set=[(X_val_use, y_val)],
                            eval_metric='multi_logloss',
                            early_stopping_rounds=30,
                            callbacks=[LightGBMPruningCallback(trial, 'multi_logloss')],
                            verbose=False
                        )
                    except TypeError:
                        model.fit(X_tr_use, y_tr)
                else:
                    model.fit(X_tr_use, y_tr)

                preds = model.predict(X_val_use)
                if optimize_metric == 'f1_weighted':
                    score = f1_score(y_val, preds, average='weighted', zero_division=0)
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

    print(f"Starting Optuna study for {class_name} with n_trials={n_trials}...")
    try:
        num_workers = max(1, min(num_cores // 2, num_cores))
    except Exception:
        num_workers = 1
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=num_workers)

    best_params = study.best_params
    # save summary
    header = not os.path.exists(summary_file)
    with open(summary_file, 'a') as fh:
        if header:
            fh.write('FeatureSet,Classifier,BestScore,BestParams\n')
        fh.write(f"{feature_name},{class_name},{study.best_value},'{json.dumps(best_params)}'\n")

    final_clf = build_final_classifier(best_params)
    elapsed = time.time() - start_time
    print(f"Optuna HPO for {class_name} completed in {elapsed:.1f}s; best_value={study.best_value:.4f}")
    return final_clf, elapsed

# --------------------
# Orchestration: supervised ML across feature sets
# --------------------
def supervised_ml(feature_sets, y):
    warnings.filterwarnings("ignore")

    classifiers = {
        "LightGBM": LGBMClassifier(verbosity=-1),
        "Logistic Regression": LogisticRegression(verbose=0),
        "Naive Bayes": MultinomialNB(),
        "Support Vector Classifier": SVC(verbose=0),
        "Random Forest Classifier": RandomForestClassifier(verbose=0),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "k-NN Classifier": KNeighborsClassifier(),
        "MLP Classifier": MLPClassifier(verbose=0),
        "Bagging Classifier": BaggingClassifier(),
        "Perceptron": Perceptron(verbose=0),
        "SGD Classifier": SGDClassifier(verbose=0),
        "Extra Trees Classifier": ExtraTreesClassifier(),
        "Ridge Classifier": RidgeClassifier(),
        "AdaBoost Classifier": AdaBoostClassifier(),
        "Gradient Boosting Classifier": GradientBoostingClassifier(verbose=0),
        "XGBoost": XGBClassifier(verbosity=0)
    }

    output_dir = f"predictions_{today_date}"
    os.makedirs(output_dir, exist_ok=True)

    results = []
    optuna_timings = []

    for feature_name, X in feature_sets.items():
        feature_hpo_time = 0.0
        is_sparse_matrix = sparse.issparse(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

        # scaling
        if is_sparse_matrix:
            scaler = StandardScaler(with_mean=False)
            try:
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            except Exception:
                # if fails, leave as is
                X_train_scaled, X_test_scaled = X_train, X_test
        else:
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

        print(f"Running HPO for feature: {feature_name}")
        best_classifiers = {}
        for class_name, classifier in classifiers.items():
            tuned_classifier, hpo_time = calculate_params_bayesian(classifier, class_name, X_train_scaled, y_train, feature_name, n_trials=100)
            best_classifiers[class_name] = tuned_classifier
            feature_hpo_time += hpo_time

        print_time_taken(feature_hpo_time, f"Optuna tuning for {feature_name}")

        # train & evaluate in parallel
        parallel_results = Parallel(n_jobs=num_cores)(
            delayed(classifier_model)(classifier, class_name, X_train_scaled, X_test_scaled, y_train, y_test)
            for class_name, classifier in best_classifiers.items()
        )

        for result, (class_name, classifier) in zip(parallel_results, best_classifiers.items()):
            result['Feature Set'] = feature_name
            results.append(result)

            predictions_df = pd.DataFrame({
                "True Labels": y_test,
                "Predicted Labels": result['Predictions']
            })
            file_name_pred = f"{class_name}_{feature_name}_predictions.csv".replace(" ", "_")
            predictions_df.to_csv(os.path.join(output_dir, file_name_pred), index=False)

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

def plot_performance_scatter(results_df):
    output_dir = f"performance_plots_{today_date}"
    os.makedirs(output_dir, exist_ok=True)

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

# --------------------
# Main script: load data, prepare features, run experiments
# --------------------
if __name__ == "__main__":
    # Load dataset robustly like in publication2.py
    dfl = pd.read_csv("Sentences_AllAgree.csv", header=None, names=['Label', 'Text'], encoding='ISO-8859-1')
    dfl = dfl.dropna().reset_index(drop=True)

    # nltk & spacy setup
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

    # Clean texts (for TF-IDF / BoW / char n-grams pipelines)
    print("Cleaning texts...")
    dfl['Cleaned_text'] = dfl['Text'].apply(lambda x: clean_text(x))

    # --- Vectorizers for sparse features
    bow_vectorizer = CountVectorizer()
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features=20000)
    tfidf_bigram_vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=20000)
    char_vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,5), max_features=20000)

    print("Computing sparse features (BoW, TF-IDF, TF-IDF 1-2gram, Char n-grams)...")
    bow_matrix = feature_extraction(dfl, bow_vectorizer)
    tfidf_matrix = feature_extraction(dfl, tfidf_vectorizer)
    tfidf_bigram_matrix = feature_extraction(dfl, tfidf_bigram_vectorizer)
    char_matrix = feature_extraction(dfl, char_vectorizer)

    # --- TF-IDF -> TruncatedSVD (LSA)
    print("Computing TF-IDF -> LSA (TruncatedSVD)...")
    svd_components = 200  # you can change
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    lsa_pipeline = make_pipeline(tfidf_vectorizer, svd, Normalizer(copy=False))
    X_lsa = lsa_pipeline.fit_transform(dfl['Cleaned_text'])  # dense (n_samples x svd_components)

    # --- Embeddings (train Word2Vec/FastText on raw token lists)
    print("Tokenizing raw text and training Word2Vec and FastText models...")
    token_lists_raw = dfl['Text'].astype(str).apply(word_tokenize).tolist()  # we use raw text tokenization for embeddings

    embedding_vector_size = 100
    w2v_model = Word2Vec(sentences=token_lists_raw, vector_size=embedding_vector_size, window=5, min_count=1, workers=max(1, num_cores - 1))
    fasttext_model = FastText(sentences=token_lists_raw, vector_size=embedding_vector_size, window=5, min_count=1, workers=max(1, num_cores - 1))

    # Attempt to load GloVe (optional)
    glove_vectors = None
    try:
        glove_vectors = gensim_api.load(f"glove-wiki-gigaword-{embedding_vector_size}")
    except Exception as e:
        print(f"Could not load GloVe vectors: {e}. Continuing without GloVe.")

    # --- Compute simple averaged embeddings (for diagnostics) and TF-IDF-weighted embeddings
    print("Computing averaged embeddings (Word2Vec) - simple mean (diagnostic).")
    averaged_w2v = compute_document_embeddings(token_lists_raw, w2v_model.wv, embedding_vector_size, lowercase_fallback=True)

    # TF-IDF-weighted averaged embeddings - we will use tfidf_vectorizer fitted earlier
    print("Fitting TF-IDF vectorizer (already fitted in LSA pipeline); computing TF-IDF-weighted averaged embeddings using Word2Vec lookup.")
    # ensure tfidf_vectorizer is fitted - it was used within lsa_pipeline; but to be safe we fit a fresh tfidf for tokenised cleaned text
    tfidf_for_weight = TfidfVectorizer(ngram_range=(1,1), max_features=20000)
    tfidf_for_weight.fit(dfl['Cleaned_text'])
    # For token lists we use cleaned tokens to match tfidf tokenization
    token_lists_cleaned = dfl['Cleaned_text'].astype(str).apply(word_tokenize).tolist()
    tfidf_weighted_w2v = compute_tfidf_weighted_embeddings(token_lists_cleaned, tfidf_for_weight, w2v_model.wv, embedding_vector_size, lowercase_fallback=True)

    # --- Build feature_sets (6 features) - choose canonical names
    feature_sets = {
        "BoW": bow_matrix,
        "TF-IDF": tfidf_matrix,
        "TF-IDF 1-2gram": tfidf_bigram_matrix,
        "Char 3-5gram": char_matrix,
        "TF-IDF + LSA": X_lsa,  # dense
        "TF-IDF-weighted-Emb": tfidf_weighted_w2v  # dense
    }

    # --- Prepare labels (map to ints)
    y = dfl['Label'].map({'negative': 0, 'neutral': 1, 'positive': 2})
    y = y.astype(int)

    # Print shapes for sanity
    print("Feature shapes:")
    print("BoW", bow_matrix.shape)
    print("TF-IDF", tfidf_matrix.shape)
    print("TF-IDF 1-2gram", tfidf_bigram_matrix.shape)
    print("Char 3-5gram", char_matrix.shape)
    print("TF-IDF + LSA", X_lsa.shape)
    print("TF-IDF-weighted-Emb", tfidf_weighted_w2v.shape)
    print("Label distribution:\n", y.value_counts())

    # --- Run supervised ML experiments (16 classifiers x 6 features)
    results_df = supervised_ml(feature_sets, y)

    # Postprocess results
    cols = ['Feature Set', 'Model name', 'F1 score', 'Time [s]']
    results_df['Time [s]'] = results_df['Time [s]'].round(2)
    sorted_df = results_df.sort_values(by=['Feature Set', 'F1 score'], ascending=[True, False])
    print(sorted_df[cols])

    results_df.to_csv(f"ml_results_{today_date}.csv", index=False)
    plot_performance_scatter(sorted_df[['Feature Set', 'Model name', 'F1 score', 'Time [s]']].copy())
    print("All done. Results saved.")
