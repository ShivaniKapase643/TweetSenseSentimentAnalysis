from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

from src.preprocessing import preprocess_text


def get_top_n_words(df: pd.DataFrame, text_col: str = "tweet", n: int = 20):
    """
    Extracts top N most frequent words from processed text in DataFrame.
    """
    if text_col not in df.columns or len(df) == 0:
        return pd.DataFrame(columns=["Word", "Frequency"])

    all_words = []
    for text in df[text_col].dropna():
        proc = preprocess_text(str(text))
        tokens = proc.split()
        all_words.extend(tokens)

    counts = Counter(all_words)
    top_words = counts.most_common(n)
    return pd.DataFrame(top_words, columns=["Word", "Frequency"])


def get_sentiment_keywords(df: pd.DataFrame, text_col: str = "tweet", sentiment_col: str = "sentiment", top_n: int = 15):
    """
    Extracts top keywords grouped by Positive, Negative, and Neutral sentiment.
    """
    results = {}
    if text_col not in df.columns or sentiment_col not in df.columns:
        return results

    df_clean = df.dropna(subset=[text_col, sentiment_col]).copy()
    sentiments = df_clean[sentiment_col].astype(str).str.capitalize().unique()

    for sent in ["Positive", "Negative", "Neutral"]:
        subset = df_clean[df_clean[sentiment_col].astype(str).str.capitalize() == sent]
        words = []
        for text in subset[text_col]:
            proc = preprocess_text(str(text))
            words.extend(proc.split())
        counts = Counter(words)
        results[sent] = pd.DataFrame(counts.most_common(top_n), columns=["Word", "Frequency"])

    return results


def generate_wordcloud_figure(text_series: pd.Series, title: str = "Word Cloud", background_color: str = "white"):
    """
    Generates a WordCloud Matplotlib figure from a series of text strings.
    Handles small datasets gracefully.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    combined_text = " ".join([preprocess_text(str(t)) for t in text_series.dropna() if str(t).strip()])
    
    if not combined_text.strip() or len(combined_text.split()) < 3:
        ax.text(0.5, 0.5, "Insufficient text data to generate Word Cloud", 
                ha='center', va='center', fontsize=12, color='gray')
        ax.axis('off')
        return fig

    if not WORDCLOUD_AVAILABLE:
        ax.text(0.5, 0.5, "wordcloud library not installed", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig

    try:
        wc = WordCloud(
            width=800,
            height=400,
            background_color=background_color,
            colormap='viridis',
            max_words=100
        ).generate(combined_text)
        
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=14, pad=10)
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not generate Word Cloud: {str(e)}", ha='center', va='center', fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    return fig
