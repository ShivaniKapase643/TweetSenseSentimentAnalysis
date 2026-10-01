import sys
import os
sys.path.insert(0, os.path.abspath("."))

import pytest
import joblib
from src.feature_extraction import load_vectorizer, VECTORIZER_PATH
from src.model_training import MODEL_PATH, METADATA_PATH


def test_model_artifacts_exist():
    assert os.path.exists(MODEL_PATH), f"Model file missing at {MODEL_PATH}"
    assert os.path.exists(VECTORIZER_PATH), f"Vectorizer file missing at {VECTORIZER_PATH}"
    assert os.path.exists(METADATA_PATH), f"Metadata file missing at {METADATA_PATH}"


def test_vectorizer_loading():
    vec = load_vectorizer(VECTORIZER_PATH)
    assert vec is not None
    assert hasattr(vec, "vocabulary_")
    assert len(vec.vocabulary_) > 0


def test_model_loading():
    model = joblib.load(MODEL_PATH)
    assert model is not None
    assert hasattr(model, "predict")
