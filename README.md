# TweetSense – Intelligent Tweet Sentiment Analysis System

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![ML Library](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> **Tagline:** Understand Opinions. Analyze Sentiments. Discover Trends.

**TweetSense** is an academic NLP (Natural Language Processing) and Machine Learning application designed to classify social media text/tweets into **Positive**, **Negative**, or **Neutral** sentiment categories.

---

## 📌 Project Overview
TweetSense provides an end-to-end explainable sentiment analysis workflow. Users can evaluate individual tweets, upload bulk CSV datasets, inspect step-by-step NLP text preprocessing, examine multi-model evaluation metrics (Logistic Regression, Naive Bayes, Linear SVM), and export PDF reports.

### Key Features
1. **Single Tweet Sentiment Analysis:** Real-time sentiment prediction with statistical model confidence scores and TF-IDF feature explainability.
2. **Batch CSV Processing:** Bulk analysis of dataset files with automatic text column detection and downloadable results.
3. **Interactive Dashboard:** Dynamic KPI metric cards, Plotly charts (bar charts, donut charts, confidence distribution).
4. **Step-by-Step Preprocessing Demo:** Educational breakdown demonstrating lowercasing, URL/@mention removal, emoji mapping, contraction expansion, tokenization, stopword filtering, and negation preservation.
5. **Multi-Model Performance Comparison:** Measured benchmark table comparing Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
6. **Word & Keyword Analysis:** Term frequency charts, sentiment-specific top keyword lists, and interactive word clouds.
7. **Downloadable PDF Reports:** Professional PDF summary report generation powered by ReportLab.
8. **No Twitter API Key Required:** Completely functional using manual entry, uploaded CSV files, or bundled offline demo datasets.

---

## 🛠️ Technology Stack
- **Programming Language:** Python 3.11
- **User Interface:** Streamlit 1.30+
- **Data Processing:** Pandas, NumPy
- **Natural Language Processing:** NLTK, scikit-learn
- **Machine Learning:** TF-IDF Vectorizer, Logistic Regression (Primary), Multinomial Naive Bayes, Linear SVM
- **Visualization:** Plotly, Matplotlib, WordCloud
- **Report Generation:** ReportLab
- **Testing:** Pytest

---

## 🏗️ System Architecture
```
User / Input Tweet / CSV File
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
   [Sentiment & Confidence] ──► Class output + Feature contribution scores
             │
             ▼
[Interactive Streamlit UI] ──► Analytics Dashboard, Batch CSV Export & PDF Report
```

---

## ⚡ Quick Start & Running Locally

### 1. Clone & Setup Environment
```bash
git clone https://github.com/your-username/TweetSense.git
cd TweetSense
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Run Pytest Suite
```bash
python -m pytest
```

### 5. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Measured Model Performance

| Model Algorithm | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|---|---|---|---|---|
| **Logistic Regression (Primary)** | **89.47%** | **90.23%** | **89.47%** | **89.44%** |
| **Linear SVM** | 89.47% | 90.23% | 89.47% | 89.44% |
| **Multinomial Naive Bayes** | 73.68% | 76.54% | 73.68% | 73.08% |

---

## 📁 Repository Structure
```
TweetSense/
│
├── app.py                      # Main Streamlit application
├── train_model.py              # CLI training script
├── run_streamlit.py            # Render production server entrypoint
│
├── data/
│   ├── tweets.csv              # Full demonstration dataset
│   └── sample_tweets.csv       # Lightweight sample dataset
│
├── models/
│   ├── sentiment_model.pkl     # Trained primary model artifact
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer artifact
│   └── model_metadata.json     # Measured model performance metadata
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Complete NLP pipeline & step-by-step breakdown
│   ├── feature_extraction.py   # TF-IDF vectorizer utilities
│   ├── model_training.py       # Multi-model training logic
│   ├── prediction.py           # Prediction & explainability engine
│   ├── evaluation.py           # Evaluation metrics logic
│   ├── visualization.py        # Plotly & Matplotlib chart generation
│   ├── text_analysis.py        # Word frequency & WordCloud tools
│   └── report_generator.py     # ReportLab PDF generator
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   ├── test_model.py
│   └── test_app_logic.py
│
├── reports/                    # Generated PDF reports output folder
├── requirements.txt
├── .gitignore
├── .python-version
├── README.md
├── DEPLOYMENT.md
├── PROJECT_DOCUMENTATION.md    # 18-Chapter Academic Documentation
├── PRESENTATION.md             # 15-Slide Presentation Deck Outline
└── VIVA_QA.md                  # 40+ Viva Questions & Answers
```

---

## ⚠️ Academic & Ethical Disclaimer
- The dataset provided (`data/tweets.csv`) is a synthetic demonstration dataset created for evaluation and testing.
- Sentiment analysis outputs reflect statistical model confidence derived from training patterns, not absolute human emotional truth.
- Sarcasm, domain-specific slang, and implicit context remain inherent challenges in statistical NLP models.

---

## 📜 License
This project is licensed under the MIT License - free for academic and educational use.
