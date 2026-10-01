import os
import json
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Add current path to sys.path
sys.path.insert(0, os.path.abspath("."))

from src.preprocessing import preprocess_text, preprocess_text_step_by_step
from src.prediction import predict_single_tweet, predict_batch_df, get_model_and_vectorizer
from src.visualization import (
    plot_sentiment_distribution_bar,
    plot_sentiment_donut_chart,
    plot_confidence_distribution,
    plot_word_frequency_bar,
    plot_confusion_matrix_heatmap,
    plot_model_comparison_bar,
    plot_sentiment_over_time,
    plot_tweet_length_distribution
)
from src.text_analysis import get_top_n_words, get_sentiment_keywords, generate_wordcloud_figure
from src.report_generator import generate_pdf_report

# Page Config
st.set_page_config(
    page_title="TweetSense – Tweet Sentiment Analysis System",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished KPI cards & styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 20px;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 18px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .kpi-card-pos { border-left-color: #2ecc71; }
    .kpi-card-neg { border-left-color: #e74c3c; }
    .kpi-card-neu { border-left-color: #3498db; }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 5px 0;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 600;
    }
    .sentiment-badge-pos {
        background-color: #d4edda;
        color: #155724;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .sentiment-badge-neg {
        background-color: #f8d7da;
        color: #721c24;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .sentiment-badge-neu {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# Load dataset and metadata safely with caching
@st.cache_data
def load_default_dataset():
    path = os.path.join("data", "tweets.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Pre-predict if needed
        if "predicted_sentiment" not in df.columns or "confidence" not in df.columns:
            df = predict_batch_df(df, text_col="tweet")
        return df
    return pd.DataFrame(columns=["tweet", "sentiment", "predicted_sentiment", "confidence"])


@st.cache_data
def load_model_metadata():
    path = os.path.join("models", "model_metadata.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


df_dataset = load_default_dataset()
metadata = load_model_metadata()

# Sidebar Navigation
st.sidebar.title("🐦 TweetSense Nav")
st.sidebar.markdown("**NLP Sentiment Analysis**")

nav_option = st.sidebar.radio(
    "Select Page:",
    [
        "🏠 Dashboard",
        "🔍 Single Tweet Analysis",
        "📂 Batch Analysis",
        "📊 Sentiment Analytics",
        "🧹 Text Preprocessing",
        "🤖 Model Performance",
        "🔤 Word Analysis",
        "📄 Reports",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("System Status: **Model Active**")
if metadata and "primary_model" in metadata:
    st.sidebar.caption(f"Model: `{metadata['primary_model']}`")
    st.sidebar.caption(f"Accuracy: `{metadata['primary_metrics']['accuracy']*100:.1f}%`")


# ==============================================================================
# 1. 🏠 DASHBOARD
# ==============================================================================
if nav_option == "🏠 Dashboard":
    st.markdown('<p class="main-title">TweetSense Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Understand Opinions. Analyze Sentiments. Discover Trends.</p>', unsafe_allow_html=True)

    if df_dataset.empty:
        st.warning("Default dataset not found or empty. Please run model training first.")
    else:
        total_tweets = len(df_dataset)
        pos_df = df_dataset[df_dataset["predicted_sentiment"] == "Positive"]
        neg_df = df_dataset[df_dataset["predicted_sentiment"] == "Negative"]
        neu_df = df_dataset[df_dataset["predicted_sentiment"] == "Neutral"]

        pos_count, neg_count, neu_count = len(pos_df), len(neg_df), len(neu_df)
        pos_pct = (pos_count / total_tweets * 100) if total_tweets > 0 else 0
        neg_pct = (neg_count / total_tweets * 100) if total_tweets > 0 else 0
        neu_pct = (neu_count / total_tweets * 100) if total_tweets > 0 else 0

        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">TOTAL TWEETS</div><div class="kpi-value">{total_tweets}</div><div>Dataset Count</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="kpi-card kpi-card-pos"><div class="kpi-label">POSITIVE TWEETS</div><div class="kpi-value">{pos_count}</div><div>{pos_pct:.1f}% of total</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="kpi-card kpi-card-neg"><div class="kpi-label">NEGATIVE TWEETS</div><div class="kpi-value">{neg_count}</div><div>{neg_pct:.1f}% of total</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="kpi-card kpi-card-neu"><div class="kpi-label">NEUTRAL TWEETS</div><div class="kpi-value">{neu_count}</div><div>{neu_pct:.1f}% of total</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Overview Charts
        c1, c2 = st.columns(2)
        with c1:
            fig_bar = plot_sentiment_distribution_bar(df_dataset)
            st.plotly_chart(fig_bar, use_container_width=True)
        with c2:
            fig_donut = plot_sentiment_donut_chart(df_dataset)
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("### 📈 Confidence Score Distribution")
        fig_conf = plot_confidence_distribution(df_dataset)
        st.plotly_chart(fig_conf, use_container_width=True)


# ==============================================================================
# 2. 🔍 SINGLE TWEET ANALYSIS
# ==============================================================================
elif nav_option == "🔍 Single Tweet Analysis":
    st.markdown('<p class="main-title">Single Tweet Sentiment Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Enter a tweet to classify its sentiment and inspect feature explainability.</p>', unsafe_allow_html=True)

    st.markdown("#### 💡 Quick Test Examples:")
    ex_col1, ex_col2, ex_col3, ex_col4 = st.columns(4)
    
    preset_text = ""
    if ex_col1.button("😊 Positive Sample"):
        preset_text = "I absolutely love this new product! The design is stunning and performance is unreal. ❤️"
    if ex_col2.button("😡 Negative Sample"):
        preset_text = "This is the worst customer service experience I have ever had. Extremely disappointed! 😡"
    if ex_col3.button("😐 Neutral Sample"):
        preset_text = "The monthly team update meeting will be held via Zoom tomorrow at 5 PM."
    if ex_col4.button("🔍 Negation Handling"):
        preset_text = "Honestly, I don't hate the new update. It's actually not bad at all."

    input_text = st.text_area("Type or paste a tweet here...", value=preset_text, height=120)

    if st.button("🚀 Analyze Sentiment", type="primary"):
        if not input_text.strip():
            st.error("Please enter a valid tweet text before analyzing.")
        else:
            with st.spinner("Analyzing tweet sentiment..."):
                res = predict_single_tweet(input_text)

            sentiment = res["predicted_sentiment"]
            confidence = res["confidence"]
            probs = res["class_probabilities"]
            explanation = res["explanation_features"]

            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown("### Predicted Sentiment:")
                if sentiment == "Positive":
                    st.markdown(f'<div class="sentiment-badge-pos">🟢 POSITIVE</div>', unsafe_allow_html=True)
                elif sentiment == "Negative":
                    st.markdown(f'<div class="sentiment-badge-neg">🔴 NEGATIVE</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="sentiment-badge-neu">🔵 NEUTRAL</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Model Confidence", f"{confidence:.1f}%")
                st.caption("ℹ️ Note: Labeled value reflects statistical model confidence, not absolute certainty.")

            with res_col2:
                st.markdown("### Class Probability Breakdown")
                for cls_name in ["Positive", "Negative", "Neutral"]:
                    p_val = probs.get(cls_name, 0.0)
                    st.write(f"**{cls_name}**: {p_val:.1f}%")
                    st.progress(min(max(p_val / 100.0, 0.0), 1.0))

            # Explainability Section
            st.markdown("---")
            st.markdown("### 🔤 Feature Explainability (Top Contributing Terms)")
            st.write(f"Processed Text: `{res['processed_text']}`")

            if explanation:
                exp_df = pd.DataFrame(explanation)
                exp_df.columns = ["Word / Feature", "TF-IDF Weight", "Model Impact Score"]
                st.dataframe(exp_df, use_container_width=True)
            else:
                st.info("No specific strong TF-IDF keywords detected for feature breakdown.")


# ==============================================================================
# 3. 📂 BATCH ANALYSIS
# ==============================================================================
elif nav_option == "📂 Batch Analysis":
    st.markdown('<p class="main-title">Batch CSV Sentiment Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload a CSV file containing tweets to perform bulk sentiment classification.</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    use_sample = st.checkbox("Or use bundled sample dataset (data/sample_tweets.csv)")

    df_batch = None
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    elif use_sample:
        sample_path = os.path.join("data", "sample_tweets.csv")
        if os.path.exists(sample_path):
            df_batch = pd.read_csv(sample_path)

    if df_batch is not None:
        st.write("### Data Preview (First 5 Rows)")
        st.dataframe(df_batch.head(), use_container_width=True)

        # Select text column
        candidate_cols = [c for c in df_batch.columns if any(k in c.lower() for k in ["tweet", "text", "content", "comment"])]
        default_col_idx = df_batch.columns.get_loc(candidate_cols[0]) if candidate_cols else 0

        selected_text_col = st.selectbox("Select Text Column:", df_batch.columns, index=default_col_idx)

        if st.button("⚡ Run Batch Analysis", type="primary"):
            with st.spinner("Processing batch tweets..."):
                res_batch = predict_batch_df(df_batch, text_col=selected_text_col)

            st.success(f"Successfully processed {len(res_batch)} tweets!")

            # Batch Summary KPIs
            b_pos = len(res_batch[res_batch["predicted_sentiment"] == "Positive"])
            b_neg = len(res_batch[res_batch["predicted_sentiment"] == "Negative"])
            b_neu = len(res_batch[res_batch["predicted_sentiment"] == "Neutral"])

            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Total Processed", len(res_batch))
            bc2.metric("Positive Tweets", b_pos)
            bc3.metric("Negative Tweets", b_neg)
            bc4.metric("Neutral Tweets", b_neu)

            # Results Table
            st.markdown("### Analyzed Batch Results")
            st.dataframe(res_batch[[selected_text_col, "processed_text", "predicted_sentiment", "confidence"]], use_container_width=True)

            # Download Button
            csv_data = res_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Analyzed CSV",
                data=csv_data,
                file_name="analyzed_tweets_results.csv",
                mime="text/csv"
            )


# ==============================================================================
# 4. 📊 SENTIMENT ANALYTICS
# ==============================================================================
elif nav_option == "📊 Sentiment Analytics":
    st.markdown('<p class="main-title">Sentiment Analytics & Filtering</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Search, filter, and analyze tweets by sentiment category.</p>', unsafe_allow_html=True)

    if df_dataset.empty:
        st.warning("No dataset loaded.")
    else:
        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            sentiment_filter = st.selectbox("Filter by Sentiment:", ["All", "Positive", "Negative", "Neutral"])
        with col_f2:
            search_query = st.text_input("Search in tweets:", "")

        df_filtered = df_dataset.copy()
        if sentiment_filter != "All":
            df_filtered = df_filtered[df_filtered["predicted_sentiment"] == sentiment_filter]

        if search_query.strip():
            df_filtered = df_filtered[df_filtered["tweet"].astype(str).str.contains(search_query, case=False, na=False)]

        st.info(f"Showing **{len(df_filtered)}** out of **{len(df_dataset)}** tweets.")

        # Interactive Table
        st.dataframe(df_filtered[["tweet", "predicted_sentiment", "confidence"]], use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Tweet Length vs Sentiment Analysis")
        fig_len = plot_tweet_length_distribution(df_dataset)
        st.plotly_chart(fig_len, use_container_width=True)

        if "date" in df_dataset.columns:
            st.markdown("---")
            st.markdown("### 📅 Sentiment Trends Over Time")
            fig_time = plot_sentiment_over_time(df_dataset, date_col="date")
            st.plotly_chart(fig_time, use_container_width=True)


# ==============================================================================
# 5. 🧹 TEXT PREPROCESSING
# ==============================================================================
elif nav_option == "🧹 Text Preprocessing":
    st.markdown('<p class="main-title">NLP Preprocessing Pipeline Demonstration</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Step-by-step breakdown of how raw tweet text is cleaned for ML feature extraction.</p>', unsafe_allow_html=True)

    sample_demo_text = st.text_input(
        "Enter sample tweet to demonstrate preprocessing:",
        value="I don't hate this! Check out https://t.co/xyz @user #awesome ❤️!!"
    )

    if sample_demo_text:
        breakdown = preprocess_text_step_by_step(sample_demo_text)

        st.markdown("### Step-by-Step Transformation:")
        
        steps_info = [
            ("1. Original Raw Input", breakdown["1_original"], "Raw tweet string as provided by user."),
            ("2. Emoji & Contraction Handling", breakdown["2_emoji_and_contractions"], "Mapped emojis (❤️ -> love) & expanded contractions (don't -> do not)."),
            ("3. Lowercasing", breakdown["3_lowercased"], "Converted all characters to lowercase."),
            ("4. URL & @Mention Removal", breakdown["4_urls_mentions_removed"], "Stripped HTTP links and Twitter handle mentions."),
            ("5. Hashtag Cleaning", breakdown["5_hashtags_cleaned"], "Preserved sentiment-bearing hashtag text (#awesome -> awesome)."),
            ("6. Punctuation & Digit Removal", breakdown["6_punctuation_digits_removed"], "Removed numbers and punctuation symbols."),
            ("7. Whitespace Cleaning", breakdown["7_whitespace_cleaned"], "Stripped extra spaces and tabs."),
            ("8. Tokenization", str(breakdown["8_tokens"]), "Split text into individual word tokens."),
            ("9. Stopword Filtering (Negation Preserved)", str(breakdown["9_stopwords_filtered"]), "Removed non-informative stopwords while preserving negation words ('not', 'no', 'never')."),
            ("10. Final Lemmatization Output", breakdown["10_lemmatized_final"], "Reduced words to base dictionary forms.")
        ]

        for title, val, desc in steps_info:
            with st.expander(title, expanded=False):
                st.code(val, language="text")
                st.caption(desc)

        st.success(f"Final Preprocessed Output: `{breakdown['10_lemmatized_final']}`")


# ==============================================================================
# 6. 🤖 MODEL PERFORMANCE
# ==============================================================================
elif nav_option == "🤖 Model Performance":
    st.markdown('<p class="main-title">Model Evaluation & Metrics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Measured performance metrics across algorithms.</p>', unsafe_allow_html=True)

    if not metadata:
        st.warning("Model metadata missing. Please run model training first.")
    else:
        p_metrics = metadata.get("primary_metrics", {})
        comp_res = metadata.get("comparison_results", {})

        # KPI Metrics Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{p_metrics.get('accuracy', 0)*100:.2f}%")
        col2.metric("Precision (Weighted)", f"{p_metrics.get('precision', 0)*100:.2f}%")
        col3.metric("Recall (Weighted)", f"{p_metrics.get('recall', 0)*100:.2f}%")
        col4.metric("F1-Score (Weighted)", f"{p_metrics.get('f1_score', 0)*100:.2f}%")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Confusion Matrix")
            cm = p_metrics.get("confusion_matrix", [[0,0,0],[0,0,0],[0,0,0]])
            fig_cm = plot_confusion_matrix_heatmap(cm)
            st.plotly_chart(fig_cm, use_container_width=True)

        with c2:
            st.markdown("### Classification Report")
            class_report = p_metrics.get("classification_report", {})
            if class_report:
                cr_df = pd.DataFrame(class_report).transpose().round(4)
                st.dataframe(cr_df, use_container_width=True)

        # Multi-Model Comparison
        st.markdown("---")
        st.markdown("### 📊 Multi-Model Performance Comparison")
        if comp_res:
            comp_rows = []
            for name, res in comp_res.items():
                comp_rows.append({
                    "Model": name,
                    "Accuracy": res["accuracy"],
                    "Precision": res["precision"],
                    "Recall": res["recall"],
                    "F1-Score": res["f1_score"]
                })
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, use_container_width=True)

            fig_comp = plot_model_comparison_bar(comp_df)
            st.plotly_chart(fig_comp, use_container_width=True)


# ==============================================================================
# 7. 🔤 WORD ANALYSIS
# ==============================================================================
elif nav_option == "🔤 Word Analysis":
    st.markdown('<p class="main-title">Word & Keyword Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Frequency distributions and top keywords by sentiment class.</p>', unsafe_allow_html=True)

    if df_dataset.empty:
        st.warning("No dataset loaded.")
    else:
        top_words_df = get_top_n_words(df_dataset, text_col="tweet", n=20)
        
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_wf = plot_word_frequency_bar(top_words_df, title="Top 15 Most Frequent Terms")
            st.plotly_chart(fig_wf, use_container_width=True)
        with c2:
            st.markdown("### ☁️ Word Cloud")
            fig_wc = generate_wordcloud_figure(df_dataset["tweet"], title="Dataset Word Cloud")
            st.pyplot(fig_wc)

        st.markdown("---")
        st.markdown("### 🔑 Sentiment-Specific Top Keywords")
        keywords_dict = get_sentiment_keywords(df_dataset, text_col="tweet", sentiment_col="predicted_sentiment")

        t_pos, t_neg, t_neu = st.tabs(["🟢 Positive Keywords", "🔴 Negative Keywords", "🔵 Neutral Keywords"])
        with t_pos:
            st.dataframe(keywords_dict.get("Positive", pd.DataFrame()), use_container_width=True)
        with t_neg:
            st.dataframe(keywords_dict.get("Negative", pd.DataFrame()), use_container_width=True)
        with t_neu:
            st.dataframe(keywords_dict.get("Neutral", pd.DataFrame()), use_container_width=True)


# ==============================================================================
# 8. 📄 REPORTS
# ==============================================================================
elif nav_option == "📄 Reports":
    st.markdown('<p class="main-title">Project & Analysis Reports</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Generate and download formal PDF project reports.</p>', unsafe_allow_html=True)

    if df_dataset.empty:
        st.warning("No dataset available to generate report.")
    else:
        total = len(df_dataset)
        pos = len(df_dataset[df_dataset["predicted_sentiment"] == "Positive"])
        neg = len(df_dataset[df_dataset["predicted_sentiment"] == "Negative"])
        neu = len(df_dataset[df_dataset["predicted_sentiment"] == "Neutral"])
        
        p_metrics = metadata.get("primary_metrics", {})

        summary_stats = {
            "total_tweets": total,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "model_name": metadata.get("primary_model", "Logistic Regression"),
            "accuracy": p_metrics.get("accuracy", "N/A"),
            "precision": p_metrics.get("precision", "N/A"),
            "recall": p_metrics.get("recall", "N/A"),
            "f1_score": p_metrics.get("f1_score", "N/A")
        }

        st.markdown("### Report Preview Summary")
        st.json(summary_stats)

        if st.button("📄 Generate PDF Report", type="primary"):
            with st.spinner("Generating PDF report via ReportLab..."):
                pdf_path = generate_pdf_report(summary_stats)
            
            st.success(f"PDF Report generated successfully at `{pdf_path}`!")
            
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name="TweetSense_Sentiment_Report.pdf",
                mime="application/pdf"
            )


# ==============================================================================
# 9. ℹ️ ABOUT PROJECT
# ==============================================================================
elif nav_option == "ℹ️ About Project":
    st.markdown('<p class="main-title">About TweetSense</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Academic Mini Project Overview, System Architecture & Ethical Limitations</p>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Project Objective
    **TweetSense** is an academic NLP application built to analyze social media text/tweets and classify sentiment into **Positive**, **Negative**, or **Neutral** categories.

    ---

    ### 🏗️ System Architecture
    ```
    User / Input Text / CSV File
                 │
                 ▼
      [Text Preprocessing] ──► Lowercasing, URL/Mention removal, Emoji mapping, Negation preservation
                 │
                 ▼
    [TF-IDF Feature Extraction] ──► Unigram & Bigram matrix representation
                 │
                 ▼
      [Machine Learning Model] ──► Logistic Regression / Naive Bayes / Linear SVM
                 │
                 ▼
       [Sentiment & Confidence] ──► Class output + Explainable feature contribution scores
                 │
                 ▼
    [Interactive Streamlit UI] ──► Analytics Dashboard, Batch CSV Export & PDF Report
    ```

    ---

    ### ⚠️ Ethical & Technical Limitations
    1. **Sarcasm & Irony:** Statistical n-gram models can struggle with sarcastic statements (e.g., *"Oh, great, another flight delay!"*).
    2. **Contextual Nuance:** Short social media posts lack wider context.
    3. **Model Confidence vs Truth:** Model confidence scores represent statistical class probability distribution, not objective emotional truth.
    4. **Domain Shift:** Models trained on general twitter dataset may require fine-tuning for specific niche domains (e.g., medical or financial text).
    """)
