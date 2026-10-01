import os
import joblib
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_text
from src.feature_extraction import load_vectorizer, VECTORIZER_PATH
from src.model_training import MODEL_PATH

_cached_model = None
_cached_vectorizer = None


def get_model_and_vectorizer(model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
    """
    Loads and caches model and vectorizer to avoid unnecessary disk reloads.
    """
    global _cached_model, _cached_vectorizer
    if _cached_model is None or _cached_vectorizer is None:
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model or Vectorizer files missing. Please run model training first.")
        _cached_model = joblib.load(model_path)
        _cached_vectorizer = joblib.load(vectorizer_path)
    return _cached_model, _cached_vectorizer


def predict_single_tweet(text: str):
    """
    Predicts sentiment and model confidence for a single tweet string.
    Returns dict with:
    - original_text
    - processed_text
    - predicted_sentiment
    - confidence (0.0 to 100.0)
    - class_probabilities (dict)
    - explanation_features (list of dicts)
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "original_text": text,
            "processed_text": "",
            "predicted_sentiment": "Neutral",
            "confidence": 0.0,
            "class_probabilities": {"Negative": 33.33, "Neutral": 33.34, "Positive": 33.33},
            "explanation_features": [],
            "error": "Empty or non-string input provided."
        }

    model, vectorizer = get_model_and_vectorizer()

    processed_text = preprocess_text(text)
    if not processed_text.strip():
        return {
            "original_text": text,
            "processed_text": "",
            "predicted_sentiment": "Neutral",
            "confidence": 50.0,
            "class_probabilities": {"Negative": 25.0, "Neutral": 50.0, "Positive": 25.0},
            "explanation_features": [],
            "note": "Text contained only stopwords or punctuation."
        }

    X_tfidf = vectorizer.transform([processed_text])
    
    # Predict probabilities if supported, fallback to decision function
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_tfidf)[0]
        classes = list(model.classes_)
        prob_dict = {cls: float(np.round(p * 100, 2)) for cls, p in zip(classes, probs)}
        max_idx = np.argmax(probs)
        predicted_sentiment = str(classes[max_idx])
        confidence = float(np.round(probs[max_idx] * 100, 2))
    elif hasattr(model, "calibrated_classifiers_"):
        probs = model.predict_proba(X_tfidf)[0]
        classes = list(model.classes_)
        prob_dict = {cls: float(np.round(p * 100, 2)) for cls, p in zip(classes, probs)}
        max_idx = np.argmax(probs)
        predicted_sentiment = str(classes[max_idx])
        confidence = float(np.round(probs[max_idx] * 100, 2))
    else:
        pred = model.predict(X_tfidf)[0]
        predicted_sentiment = str(pred)
        confidence = 100.0
        prob_dict = {predicted_sentiment: 100.0}

    # Explainability analysis
    explanation_features = explain_prediction(model, vectorizer, X_tfidf, predicted_sentiment)

    return {
        "original_text": text,
        "processed_text": processed_text,
        "predicted_sentiment": predicted_sentiment,
        "confidence": confidence,
        "class_probabilities": prob_dict,
        "explanation_features": explanation_features
    }


def explain_prediction(model, vectorizer, X_tfidf, predicted_class):
    """
    Extracts top sentiment-contributing words/features based on TF-IDF weights
    and model coefficients (for Logistic Regression / Linear models).
    """
    explanation = []
    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
        row = X_tfidf.toarray()[0]
        active_indices = np.where(row > 0)[0]

        if len(active_indices) == 0:
            return explanation

        active_features = feature_names[active_indices]
        active_tfidf_weights = row[active_indices]

        # Extract coefficients if model is LogisticRegression or linear
        if hasattr(model, "coef_"):
            classes = list(model.classes_)
            if predicted_class in classes:
                class_idx = classes.index(predicted_class)
                # Handle binary vs multiclass coef shape
                if len(classes) == 2 and model.coef_.shape[0] == 1:
                    coefs = model.coef_[0] if class_idx == 1 else -model.coef_[0]
                else:
                    coefs = model.coef_[class_idx]
                
                impact_scores = active_tfidf_weights * coefs[active_indices]
                
                # Sort features by highest impact score
                sorted_idx = np.argsort(impact_scores)[::-1]
                for i in sorted_idx:
                    explanation.append({
                        "word": active_features[i],
                        "tfidf_weight": round(float(active_tfidf_weights[i]), 4),
                        "impact_score": round(float(impact_scores[i]), 4)
                    })
        else:
            # Fallback for non-linear models: sort by TF-IDF weight alone
            sorted_idx = np.argsort(active_tfidf_weights)[::-1]
            for i in sorted_idx:
                explanation.append({
                    "word": active_features[i],
                    "tfidf_weight": round(float(active_tfidf_weights[i]), 4),
                    "impact_score": round(float(active_tfidf_weights[i]), 4)
                })

    except Exception:
        pass

    return explanation[:10]  # Return top 10 features


def predict_batch_df(df: pd.DataFrame, text_col: str):
    """
    Batch predicts sentiment and confidence for a DataFrame containing a text column.
    """
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")

    model, vectorizer = get_model_and_vectorizer()

    # Preprocess texts
    df_result = df.copy()
    raw_texts = df_result[text_col].fillna("").astype(str).tolist()
    processed_texts = [preprocess_text(t) for t in raw_texts]
    df_result["processed_text"] = processed_texts

    # Handle empty processed texts safely
    non_empty_mask = [len(t.strip()) > 0 for t in processed_texts]

    predicted_sentiments = ["Neutral"] * len(df_result)
    confidences = [50.0] * len(df_result)

    if any(non_empty_mask):
        valid_indices = [i for i, valid in enumerate(non_empty_mask) if valid]
        valid_texts = [processed_texts[i] for i in valid_indices]

        X_tfidf = vectorizer.transform(valid_texts)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_tfidf)
            classes = list(model.classes_)
            max_indices = np.argmax(probs, axis=1)

            for idx, orig_i in enumerate(valid_indices):
                pred_cls = classes[max_indices[idx]]
                conf = float(np.round(probs[idx][max_indices[idx]] * 100, 2))
                predicted_sentiments[orig_i] = pred_cls
                confidences[orig_i] = conf
        else:
            preds = model.predict(X_tfidf)
            for idx, orig_i in enumerate(valid_indices):
                predicted_sentiments[orig_i] = str(preds[idx])
                confidences[orig_i] = 100.0

    df_result["predicted_sentiment"] = predicted_sentiments
    df_result["confidence"] = confidences

    return df_result
