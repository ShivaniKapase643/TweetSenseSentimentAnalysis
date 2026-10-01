import sys
import os
sys.path.insert(0, os.path.abspath("."))

import pytest
import pandas as pd
from src.evaluation import evaluate_model_performance


def test_evaluate_model_performance():
    y_true = ["Positive", "Negative", "Neutral", "Positive"]
    y_pred = ["Positive", "Negative", "Neutral", "Negative"]
    metrics = evaluate_model_performance(y_true, y_pred)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] == 0.75
