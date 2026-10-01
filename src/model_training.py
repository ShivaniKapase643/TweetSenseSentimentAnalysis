import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.preprocessing import preprocess_text
from src.feature_extraction import fit_and_save_vectorizer
from src.evaluation import evaluate_model_performance

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


def train_and_compare_models(df, text_col="tweet", label_col="sentiment"):
    """
    Trains multiple models (Logistic Regression, Naive Bayes, Linear SVM),
    evaluates them on a stratified test set, selects the best performing primary model,
    saves artifacts (model, vectorizer, metadata), and returns training results summary.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Clean missing / invalid records
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str).str.strip().str.capitalize()

    # Normalize labels to standard set (Positive, Negative, Neutral)
    valid_map = {"Positive": "Positive", "Negative": "Negative", "Neutral": "Neutral"}
    df["clean_sentiment"] = df[label_col].map(valid_map)
    df = df.dropna(subset=["clean_sentiment"]).copy()

    if len(df) < 10:
        raise ValueError("Dataset is too small to train a machine learning model. Need at least 10 samples.")

    # Step 1: Preprocess Text
    df["processed_text"] = df[text_col].apply(preprocess_text)
    # Remove empty processed strings
    df = df[df["processed_text"].str.strip() != ""].copy()

    # Step 2: Stratified Train/Test Split
    X = df["processed_text"]
    y = df["clean_sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 3: TF-IDF Feature Extraction
    vectorizer, X_train_tfidf = fit_and_save_vectorizer(X_train, save_path=VECTORIZER_PATH)
    X_test_tfidf = vectorizer.transform(X_test)

    # Step 4: Define candidate models
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0, class_weight='balanced'),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Linear SVM": CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000, C=1.0), cv=3)
    }

    comparison_results = {}
    best_f1 = -1.0
    best_model_name = "Logistic Regression"
    best_model_obj = None
    best_metrics = None

    # Step 5: Train and Evaluate Models
    for name, model in candidate_models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        metrics = evaluate_model_performance(y_test, y_pred)
        comparison_results[name] = metrics

        # Select model based on F1-score
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = name
            best_model_obj = model
            best_metrics = metrics

    # Save primary model (Logistic Regression is enforced as default primary if close, or best)
    # To keep default explainability stable, if Logistic Regression is top tier, select it
    if "Logistic Regression" in candidate_models and comparison_results["Logistic Regression"]["f1_score"] >= (best_f1 - 0.05):
        primary_model_name = "Logistic Regression"
        primary_model_obj = candidate_models["Logistic Regression"]
        primary_metrics = comparison_results["Logistic Regression"]
    else:
        primary_model_name = best_model_name
        primary_model_obj = best_model_obj
        primary_metrics = best_metrics

    # Save trained model object
    joblib.dump(primary_model_obj, MODEL_PATH)

    # Prepare metadata JSON
    metadata = {
        "primary_model": primary_model_name,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "total_samples": len(df),
        "class_distribution": df["clean_sentiment"].value_counts().to_dict(),
        "primary_metrics": primary_metrics,
        "comparison_results": comparison_results,
        "vectorizer_vocab_size": len(vectorizer.vocabulary_)
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    return primary_model_name, primary_metrics, comparison_results, metadata
