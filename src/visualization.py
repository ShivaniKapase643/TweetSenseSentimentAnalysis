import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Color palette for consistent professional UI
COLOR_MAP = {
    "Positive": "#2ecc71",  # Green
    "Negative": "#e74c3c",  # Red
    "Neutral": "#3498db"   # Blue
}


def plot_sentiment_distribution_bar(df: pd.DataFrame, sentiment_col: str = "predicted_sentiment"):
    """
    Creates an interactive Plotly bar chart showing tweet count by sentiment class.
    """
    counts = df[sentiment_col].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]

    fig = px.bar(
        counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map=COLOR_MAP,
        text="Count",
        title="Sentiment Distribution (Count)",
        template="plotly_white"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="Sentiment Class", yaxis_title="Number of Tweets", height=380)
    return fig


def plot_sentiment_donut_chart(df: pd.DataFrame, sentiment_col: str = "predicted_sentiment"):
    """
    Creates an interactive Plotly donut chart showing sentiment percentages.
    """
    counts = df[sentiment_col].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        color="Sentiment",
        color_discrete_map=COLOR_MAP,
        hole=0.45,
        title="Sentiment Proportion (%)",
        template="plotly_white"
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(height=380)
    return fig


def plot_confidence_distribution(df: pd.DataFrame, confidence_col: str = "confidence", sentiment_col: str = "predicted_sentiment"):
    """
    Creates a histogram showing model confidence score distribution per sentiment class.
    """
    fig = px.histogram(
        df,
        x=confidence_col,
        color=sentiment_col,
        color_discrete_map=COLOR_MAP,
        barmode="overlay",
        nbins=20,
        title="Model Confidence Distribution (%)",
        labels={confidence_col: "Confidence (%)", sentiment_col: "Sentiment"},
        template="plotly_white"
    )
    fig.update_layout(yaxis_title="Tweet Count", height=380)
    return fig


def plot_word_frequency_bar(words_df: pd.DataFrame, title: str = "Top Frequent Words"):
    """
    Creates a horizontal bar chart of top frequent words.
    """
    if words_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No word data available", showarrow=False, font=dict(size=14))
        return fig

    fig = px.bar(
        words_df.head(15),
        x="Frequency",
        y="Word",
        orientation="h",
        title=title,
        color="Frequency",
        color_continuous_scale="Blues",
        template="plotly_white"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=420)
    return fig


def plot_confusion_matrix_heatmap(cm, labels=["Negative", "Neutral", "Positive"]):
    """
    Creates an interactive Plotly heatmap for the Confusion Matrix.
    """
    cm_arr = np.array(cm)
    fig = px.imshow(
        cm_arr,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        text_auto=True,
        title="Confusion Matrix",
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        template="plotly_white"
    )
    fig.update_layout(height=420)
    return fig


def plot_model_comparison_bar(comparison_df: pd.DataFrame):
    """
    Creates a multi-bar chart comparing accuracy, precision, recall, and f1-score across models.
    """
    fig = px.bar(
        comparison_df,
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1-Score"],
        barmode="group",
        title="Model Performance Metric Comparison",
        labels={"value": "Score (0.0 to 1.0)", "variable": "Metric"},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(yaxis_range=[0, 1.05], height=420)
    return fig


def plot_sentiment_over_time(df: pd.DataFrame, date_col: str, sentiment_col: str = "predicted_sentiment"):
    """
    Plots daily sentiment count over time if a date column exists.
    """
    df_copy = df.copy()
    try:
        df_copy["date_parsed"] = pd.to_datetime(df_copy[date_col]).dt.date
    except Exception:
        fig = go.Figure()
        fig.add_annotation(text="Could not parse date column.", showarrow=False)
        return fig

    daily_sentiment = df_copy.groupby(["date_parsed", sentiment_col]).size().reset_index(name="Count")

    fig = px.line(
        daily_sentiment,
        x="date_parsed",
        y="Count",
        color=sentiment_col,
        color_discrete_map=COLOR_MAP,
        title="Sentiment Trends Over Time",
        labels={"date_parsed": "Date", "Count": "Tweet Count", sentiment_col: "Sentiment"},
        template="plotly_white",
        markers=True
    )
    fig.update_layout(height=400)
    return fig


def plot_tweet_length_distribution(df: pd.DataFrame, text_col: str = "tweet", sentiment_col: str = "predicted_sentiment"):
    """
    Plots tweet length distribution by sentiment.
    """
    df_copy = df.copy()
    df_copy["char_length"] = df_copy[text_col].astype(str).str.len()

    fig = px.box(
        df_copy,
        x=sentiment_col,
        y="char_length",
        color=sentiment_col,
        color_discrete_map=COLOR_MAP,
        title="Tweet Character Length by Sentiment",
        labels={sentiment_col: "Sentiment", "char_length": "Character Count"},
        template="plotly_white"
    )
    fig.update_layout(showlegend=False, height=380)
    return fig
