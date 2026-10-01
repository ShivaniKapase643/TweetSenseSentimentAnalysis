# TweetSense: Intelligent Tweet Sentiment Analysis System
## Presentation Deck Outline (15 Slides)

---

### Slide 1: Title Slide
- **Title:** TweetSense – Intelligent Tweet Sentiment Analysis System
- **Subtitle:** Understand Opinions. Analyze Sentiments. Discover Trends.
- **Presenter:** [Student Name / Team Name]
- **Course / Degree:** B.Tech / B.E. / M.Tech Computer Science & Engineering
- **Domain:** Natural Language Processing (NLP) & Machine Learning

---

### Slide 2: Introduction
- Social media platforms host millions of public opinions daily.
- Sentiment analysis automates the extraction of public mood and brand perception.
- **TweetSense** classifies tweets into **Positive**, **Negative**, or **Neutral** sentiment classes using an explainable ML framework.

---

### Slide 3: Problem Statement
- **Volume & Velocity:** Manual analysis of social media text is impossible at scale.
- **Text Noise:** Social posts contain emojis, @mentions, URLs, slang, and contractions.
- **Negation Pitfall:** Standard NLP stopword removal strips words like *"not"*, turning *"not good"* into *"good"*, corrupting accuracy.

---

### Slide 4: Project Objectives
1. Build an end-to-end NLP preprocessing pipeline with negation preservation.
2. Extract unigram and bigram features using TF-IDF vectorization.
3. Benchmark Logistic Regression, Naive Bayes, and Linear SVM models.
4. Develop an interactive Streamlit dashboard for real-time and batch analysis.
5. Provide feature-level explainability and downloadable PDF reports.

---

### Slide 5: Existing System vs Proposed System
| Feature | Existing Systems | Proposed TweetSense |
|---|---|---|
| **API Dependency** | Requires paid/rate-limited Twitter API | Completely API-free (Offline/CSV) |
| **Negation Handling** | Often strips negations | Preserves negations (`not`, `no`, `never`) |
| **Explainability** | Black-box output | Word-level feature impact scores |
| **Reporting** | Static UI | Exportable CSV & ReportLab PDF |

---

### Slide 6: System Architecture
- **Data Layer:** Synthetic/Upload CSV Dataset -> Text Column Detection
- **NLP Layer:** Preprocessing Engine -> Tokenizer -> TF-IDF Vectorizer
- **ML Layer:** Logistic Regression / Naive Bayes / Linear SVM Classifier
- **Presentation Layer:** Streamlit Dashboard -> Plotly Charts -> ReportLab PDF Generator

---

### Slide 7: Preprocessing Pipeline
1. **Emoji Translation:** Mapped emojis (❤️ -> `love`, 😡 -> `angry`) to text tokens.
2. **Contraction Expansion:** `don't` -> `do not`, `can't` -> `cannot`.
3. **Cleaning:** Lowercasing, URL removal, @mention stripping, hashtag text preservation.
4. **Negation-Aware Filtering:** Custom stopword list keeping negation words intact.
5. **Lemmatization:** Base dictionary form conversion via WordNet.

---

### Slide 8: Feature Extraction – TF-IDF
- **Term Frequency - Inverse Document Frequency (TF-IDF)** converts text strings into numerical matrices.
- **N-gram Range (1, 2):** Captures single words and two-word combinations (*"not bad"*, *"very happy"*).
- **Vocabulary Size:** Limited to top 5,000 features to prevent sparsity and overfitting.

---

### Slide 9: Machine Learning Methodology
- **Primary Model:** Logistic Regression (Balanced class weights, Softmax probability output).
- **Benchmark Models:**
  1. **Multinomial Naive Bayes:** Probabilistic baseline.
  2. **Linear SVM:** Calibrated Support Vector Classifier.
- **Split Strategy:** Stratified 80% Training / 20% Testing split.

---

### Slide 10: Streamlit Dashboard Features
- **🏠 Dashboard:** Real-time KPI summary cards & Plotly donut charts.
- **🔍 Single Tweet Analysis:** Real-time text classification with confidence gauge & word impact breakdown.
- **📂 Batch Analysis:** Bulk CSV upload, auto-column detection, and result download.
- **🧹 Preprocessing Demo:** Educational 10-step text transformation viewer.

---

### Slide 11: Measured Results & Performance
- **Primary Model:** Logistic Regression
- **Accuracy:** **89.47%**
- **Precision:** **90.23%**
- **Recall:** **89.47%**
- **F1-Score:** **89.44%**

**Comparison Benchmark:**
- Logistic Regression: F1 = 0.8944
- Linear SVM: F1 = 0.8944
- Naive Bayes: F1 = 0.7308

---

### Slide 12: System Demonstration / Screenshots
- *[Screenshot: Dashboard KPI Cards & Distribution Donut Chart]*
- *[Screenshot: Single Tweet Sentiment Analysis & Feature Explainability]*
- *[Screenshot: Step-by-Step Preprocessing Demonstration]*
- *[Screenshot: Multi-Model Benchmark Comparison Table]*

---

### Slide 13: Testing & Quality Assurance
- **Framework:** Pytest unit testing suite (`tests/`).
- **Test Coverage:**
  - `test_preprocessing.py`: Regex cleaning & negation preservation.
  - `test_prediction.py`: Confidence boundaries & batch prediction.
  - `test_model.py`: Joblib serialization & artifact integrity.
- **Test Result:** `14 passed in 15.42s`.

---

### Slide 14: Limitations & Ethical Considerations
- **Sarcasm Detection:** Sarcastic tweets require context beyond standard n-grams.
- **Domain Adaptability:** Models trained on general text may need retraining for niche domains.
- **Confidence vs Truth:** Model confidence represents statistical probability, not absolute emotional truth.

---

### Slide 15: Conclusion & Q&A
- **Summary:** TweetSense successfully delivers a complete, explainable, and highly accurate NLP sentiment analysis platform.
- **Key Takeaway:** Combining proper negation-aware preprocessing with TF-IDF and Logistic Regression produces an efficient 89.4% F1-score model.
- **Thank You!** Questions & Answers.
