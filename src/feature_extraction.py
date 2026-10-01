import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")


def build_tfidf_vectorizer(max_features=5000, ngram_range=(1, 2), min_df=1, max_df=0.95):
    """
    Creates and returns an unfitted TfidfVectorizer configured for tweet sentiment analysis.
    Supports unigrams and bigrams for capturing phrases like 'not good' or 'very happy'.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        token_pattern=r'\b\w+\b'
    )
    return vectorizer


def fit_and_save_vectorizer(corpus, save_path=VECTORIZER_PATH, max_features=5000, ngram_range=(1, 2)):
    """
    Fits TfidfVectorizer on corpus and saves it to disk using joblib.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vectorizer = build_tfidf_vectorizer(max_features=max_features, ngram_range=ngram_range)
    X_tfidf = vectorizer.fit_transform(corpus)
    joblib.dump(vectorizer, save_path)
    return vectorizer, X_tfidf


def load_vectorizer(load_path=VECTORIZER_PATH):
    """
    Loads a fitted TfidfVectorizer from disk.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Vectorizer file not found at {load_path}. Please run model training first.")
    vectorizer = joblib.load(load_path)
    return vectorizer


def transform_text(text_list, vectorizer):
    """
    Transforms a list of preprocessed text strings into a TF-IDF matrix using loaded vectorizer.
    """
    if isinstance(text_list, str):
        text_list = [text_list]
    return vectorizer.transform(text_list)
