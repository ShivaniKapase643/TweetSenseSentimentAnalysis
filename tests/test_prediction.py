import sys
import os
sys.path.insert(0, os.path.abspath("."))

import pytest
import pandas as pd
from src.prediction import predict_single_tweet, predict_batch_df


def test_predict_single_tweet_positive():
    res = predict_single_tweet("I absolutely love this product! Amazing experience.")
    assert res["predicted_sentiment"] in ["Positive", "Negative", "Neutral"]
    assert "confidence" in res
    assert 0.0 <= res["confidence"] <= 100.0
    assert "class_probabilities" in res


def test_predict_single_tweet_negative():
    res = predict_single_tweet("This is terrible and completely disappointing. Worst ever.")
    assert res["predicted_sentiment"] in ["Positive", "Negative", "Neutral"]
    assert res["confidence"] >= 0.0


def test_predict_single_tweet_neutral():
    res = predict_single_tweet("The meeting is scheduled for 5 PM in room 204.")
    assert res["predicted_sentiment"] in ["Positive", "Negative", "Neutral"]


def test_predict_batch_df():
    df = pd.DataFrame({
        "tweet": [
            "Great job team! Highly recommended.",
            "Horrible product, broke immediately.",
            "The library opens at 8 AM."
        ]
    })
    res_df = predict_batch_df(df, text_col="tweet")
    assert "predicted_sentiment" in res_df.columns
    assert "confidence" in res_df.columns
    assert len(res_df) == 3
