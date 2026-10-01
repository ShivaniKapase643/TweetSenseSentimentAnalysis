import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

SENTIMENT_LABELS = ['Negative', 'Neutral', 'Positive']


def evaluate_model_performance(y_true, y_pred, labels=SENTIMENT_LABELS):
    """
    Evaluates classification performance across standard NLP metrics.
    Returns a dictionary of metrics, confusion matrix, and classification report.
    """
    # Filter available labels in y_true/y_pred
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Class-wise metrics
    class_report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report_dict,
        "labels": labels
    }
    return metrics


def format_classification_report_df(class_report_dict):
    """
    Converts sklearn classification report dictionary to a clean pandas DataFrame.
    """
    df = pd.DataFrame(class_report_dict).transpose()
    df = df.round(4)
    return df
