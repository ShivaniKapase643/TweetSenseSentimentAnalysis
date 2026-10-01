import sys
import os
sys.path.insert(0, os.path.abspath("."))

import pytest
from src.preprocessing import preprocess_text, preprocess_text_step_by_step


def test_url_and_mention_removal():
    raw_text = "Check out this link https://example.com and follow @user123 for updates!"
    processed = preprocess_text(raw_text)
    assert "https" not in processed
    assert "example" not in processed
    assert "@user123" not in processed
    assert "user123" not in processed


def test_hashtag_handling():
    raw_text = "Loving the new release! #awesome #python"
    processed = preprocess_text(raw_text)
    assert "awesome" in processed
    assert "python" in processed


def test_negation_preservation():
    # Negation words like 'not', 'no' MUST be preserved
    raw_text = "This movie is not good at all"
    processed = preprocess_text(raw_text)
    assert "not" in processed
    assert "good" in processed


def test_emoji_handling():
    raw_text = "I love this app ❤️😍"
    processed = preprocess_text(raw_text)
    assert "love" in processed


def test_empty_and_invalid_input():
    assert preprocess_text("") == ""
    assert preprocess_text(None) == ""
    assert preprocess_text("   ") == ""


def test_step_by_step_breakdown():
    raw_text = "Not bad! Check https://test.com @admin #great"
    breakdown = preprocess_text_step_by_step(raw_text)
    assert "1_original" in breakdown
    assert "10_lemmatized_final" in breakdown
    assert breakdown["1_original"] == raw_text
