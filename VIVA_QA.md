# TweetSense: Viva Voce Questions & Answers
## Comprehensive 40+ Questions for Academic Examination

---

### Part 1: NLP & Text Preprocessing (Q1 – Q10)

#### Q1: What is Natural Language Processing (NLP)?
**Answer:** Natural Language Processing (NLP) is a branch of artificial intelligence that enables computers to understand, interpret, process, and generate human language in text or speech format.

#### Q2: What is Sentiment Analysis?
**Answer:** Sentiment Analysis (or opinion mining) is an NLP technique used to determine the emotional tone behind a body of text, categorizing it into classes such as Positive, Negative, or Neutral.

#### Q3: What is Tokenization?
**Answer:** Tokenization is the process of breaking down a continuous string of text into smaller units called tokens (words, subwords, or punctuation symbols).

#### Q4: What are Stopwords? Why do we remove them?
**Answer:** Stopwords are high-frequency words in a language (such as *'is'*, *'the'*, *'at'*, *'which'*) that carry minimal domain-specific semantic meaning. Removing them reduces vocabulary size and computational noise.

#### Q5: Why is standard stopword removal dangerous for sentiment analysis?
**Answer:** Standard stopword lists include negation words such as *'not'*, *'no'*, *'never'*, and *'cannot'*. Stripping these words converts *"not good"* into *"good"*, completely reversing the sentiment. In TweetSense, we explicitly preserve negation words.

#### Q6: What is Lemmatization? How does it differ from Stemming?
**Answer:**
- **Stemming** chops off word suffixes using heuristic rules (e.g., *"caring"* -> *"car"*), often producing non-dictionary words.
- **Lemmatization** uses vocabulary and morphological analysis to reduce words to their actual dictionary base form (e.g., *"caring"* -> *"care"*).

#### Q7: How does TweetSense handle emojis in tweets?
**Answer:** TweetSense uses an emoji translation map to convert sentiment-bearing emojis into textual descriptors (e.g., ❤️ -> `love`, 😡 -> `angry`, 😊 -> `happy`) before punctuation removal.

#### Q8: What is Contraction Expansion?
**Answer:** Contraction expansion replaces abbreviated words (e.g., *"don't"* -> *"do not"*, *"can't"* -> *"cannot"*) to standardize text and ensure negations are properly parsed.

#### Q9: How are @mentions and URLs handled in social media text?
**Answer:** Regular expressions (regex) are used to strip URLs (`https://...`) and Twitter handles (`@username`) because specific usernames and URLs do not carry generalizable sentiment information.

#### Q10: How are hashtags processed?
**Answer:** The `#` symbol is removed while preserving the underlying word (e.g., `#awesome` -> `awesome`) so sentiment-bearing words inside hashtags are retained.

---

### Part 2: Feature Extraction & Vectorization (Q11 – Q18)

#### Q11: What is Bag-of-Words (BoW)?
**Answer:** Bag-of-Words is a simplification model where text is represented as a multiset of its words, disregarding grammar and word order but keeping multiplicity.

#### Q12: What is TF-IDF?
**Answer:** TF-IDF stands for **Term Frequency - Inverse Document Frequency**. It evaluates how important a word is to a document in a collection or corpus:
$$\text{TF-IDF} = \text{TF}(t, d) \times \text{IDF}(t, D)$$

#### Q13: What is the formula for Term Frequency (TF)?
**Answer:** Term Frequency measures how frequently a term $t$ appears in document $d$:
$$\text{TF}(t, d) = \frac{\text{Count of } t \text{ in } d}{\text{Total words in } d}$$

#### Q14: What is Inverse Document Frequency (IDF)?
**Answer:** IDF measures how rare or common a word is across the entire corpus $D$:
$$\text{IDF}(t, D) = \log\left(\frac{\text{Total documents } N}{\text{Documents containing term } t}\right)$$

#### Q15: What are N-grams? Why use unigrams and bigrams?
**Answer:** An N-gram is a contiguous sequence of $N$ items from a text. Unigrams are single words (*"good"*), while bigrams are two-word pairs (*"not good"*). Using bigrams allows the model to capture multi-word sentiment expressions.

#### Q16: What does `max_features=5000` do in TfidfVectorizer?
**Answer:** It restricts the TF-IDF feature space to the top 5,000 most frequent terms across the corpus, reducing dimensionality and preventing memory overhead.

#### Q17: What does `sublinear_tf` mean?
**Answer:** Sublinear TF replaces term frequency $count$ with $1 + \log(count)$, diminishing the impact of words that appear repeatedly in a single document.

#### Q18: Why is TF-IDF preferred over raw word counts?
**Answer:** Raw counts penalize long documents and overemphasize common words. TF-IDF downweights common words that appear across all documents and highlights distinctive sentiment-bearing words.

---

### Part 3: Machine Learning Algorithms (Q19 – Q27)

#### Q19: Which primary machine learning model is used in TweetSense?
**Answer:** **Logistic Regression** is used as the primary classification model due to its high accuracy, fast inference speed, and explainability.

#### Q20: How does Logistic Regression work for multiclass classification?
**Answer:** It uses the **Softmax function** (multinomial logistic regression) to calculate a probability distribution across the three classes (Positive, Negative, Neutral).

#### Q21: What is Multinomial Naive Bayes?
**Answer:** Multinomial Naive Bayes is a probabilistic classifier based on Bayes' Theorem with the "naive" assumption of feature independence given the class label.

#### Q22: What is a Linear SVM?
**Answer:** A Linear Support Vector Machine finds the optimal hyperplanes that maximize the margin of separation between sentiment classes in feature space.

#### Q23: How does TweetSense obtain probability estimates from Linear SVM?
**Answer:** Linear SVM does not natively output probabilities. TweetSense wraps Linear SVM inside `CalibratedClassifierCV` using Platt scaling (sigmoid calibration).

#### Q24: What is Train/Test Split? Why use Stratified split?
**Answer:** Train/Test split divides data into training (80%) and test (20%) sets. A **Stratified** split ensures that each class ratio (Positive, Negative, Neutral) is identical in both train and test splits.

#### Q25: What is Overfitting? How is it prevented?
**Answer:** Overfitting occurs when a model learns training data noise instead of general patterns. It is prevented using regularization parameters ($C=1.0$), TF-IDF `max_features` constraints, and cross-validation.

#### Q26: What is Underfitting?
**Answer:** Underfitting occurs when a model is too simple to capture the underlying structure of the data, resulting in poor performance on both training and test data.

#### Q27: How does TweetSense explain predictions?
**Answer:** TweetSense calculates feature impact scores by multiplying the TF-IDF weight of active words by the model's learned class coefficient:
$$\text{Impact Score} = \text{TF-IDF Weight} \times \text{Model Coefficient}$$

---

### Part 4: Model Evaluation Metrics (Q28 – Q34)

#### Q28: What is Accuracy?
**Answer:** Accuracy is the ratio of correctly predicted instances to total instances:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

#### Q29: What is Precision?
**Answer:** Precision measures the proportion of correctly identified positive predictions out of all positive predictions made:
$$\text{Precision} = \frac{TP}{TP + FP}$$

#### Q30: What is Recall?
**Answer:** Recall (Sensitivity) measures the proportion of actual positives that were correctly identified:
$$\text{Recall} = \frac{TP}{TP + FN}$$

#### Q31: What is the F1-Score?
**Answer:** F1-score is the harmonic mean of Precision and Recall:
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

#### Q32: What is a Confusion Matrix?
**Answer:** A Confusion Matrix is a tabular summary of prediction results showing True Positives, True Negatives, False Positives, and False Negatives for each class.

#### Q33: What achieved metrics did TweetSense record?
**Answer:**
- **Accuracy:** 89.47%
- **Precision:** 90.23%
- **Recall:** 89.47%
- **F1-Score:** 89.44%

#### Q34: What is Weighted Average metric?
**Answer:** A weighted average calculates metrics for each class individually and averages them weighted by the number of true instances (support) for each class.

---

### Part 5: Application & Deployment (Q35 – Q40+)

#### Q35: What library is used for the web UI?
**Answer:** **Streamlit**, an open-source Python framework used to build interactive data web applications.

#### Q36: How are model files saved and loaded?
**Answer:** Model objects (`sentiment_model.pkl`) and TF-IDF vectorizers (`tfidf_vectorizer.pkl`) are serialized to disk using `joblib`.

#### Q37: How does TweetSense generate PDF reports?
**Answer:** TweetSense uses **ReportLab** to build structured PDF documents containing summary statistics, model metrics, and project metadata.

#### Q38: Why is TweetSense independent of Twitter API?
**Answer:** Twitter/X API access is rate-limited and requires paid developer credentials. TweetSense operates offline using user-entered text, uploaded CSV files, or bundled demo datasets.

#### Q39: How is TweetSense configured for Render deployment?
**Answer:** TweetSense includes `run_streamlit.py`, which binds Streamlit to `0.0.0.0` and reads Render's dynamic `$PORT` environment variable.

#### Q40: What are the main ethical limitations of TweetSense?
**Answer:** Sarcasm detection challenges, missing domain context, potential training data bias, and the fact that model confidence represents statistical probability rather than objective truth.
