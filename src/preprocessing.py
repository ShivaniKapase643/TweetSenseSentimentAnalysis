import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Robust NLTK data downloader
def ensure_nltk_resources():
    resources = ['stopwords', 'wordnet', 'punkt', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

ensure_nltk_resources()

# Initialize Lemmatizer and Stopwords
try:
    lemmatizer = WordNetLemmatizer()
    base_stopwords = set(stopwords.words('english'))
except Exception:
    lemmatizer = None
    base_stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
        "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
        'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
        'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
        'off', 'over', 'under', 'again', 'further', 'then', 'once'
    }

# CRITICAL: Preserve negation words so "not good" does NOT become "good"
NEGATION_WORDS = {
    'no', 'not', 'nor', 'neither', 'never', 'none', 'nobody', 'nothing',
    'nowhere', 'cannot', 'can\'t', 'cant', 'couldn\'t', 'couldnt', 'didn\'t',
    'didnt', 'doesn\'t', 'doesnt', 'don\'t', 'dont', 'hadn\'t', 'hadnt',
    'hasn\'t', 'hasnt', 'haven\'t', 'havent', 'isn\'t', 'isnt', 'mightn\'t',
    'mustn\'t', 'needn\'t', 'shouldn\'t', 'shouldnt', 'wasn\'t', 'wasnt',
    'weren\'t', 'werent', 'won\'t', 'wont', 'wouldn\'t', 'wouldnt', 'without', 'lack'
}

CUSTOM_STOPWORDS = base_stopwords - NEGATION_WORDS

# Emoji text map for key sentiment-bearing emojis
EMOJI_MAP = {
    '❤️': ' love ', '😍': ' love ', '🥰': ' love ', '😊': ' happy ', '😄': ' happy ',
    '😁': ' happy ', '👍': ' good ', '🙌': ' great ', '🎉': ' celebrate ', '🔥': ' awesome ',
    '✨': ' special ', '💯': ' perfect ', '👏': ' praise ', '😃': ' glad ', '😀': ' glad ',
    '😭': ' sad cry ', '😢': ' sad ', '💔': ' heartbroken ', '😞': ' disappointed ',
    '😒': ' annoyed ', '😡': ' angry ', '🤬': ' furious ', '👎': ' bad ', '🤮': ' disgusting ',
    '💩': ' terrible ', '😩': ' exhausted ', '😩': ' upset ', '🙄': ' sarcastic '
}


def replace_emojis(text: str) -> str:
    """Replaces common emojis with textual sentiment descriptors."""
    if not isinstance(text, str):
        return ""
    for emoji, text_rep in EMOJI_MAP.items():
        text = text.replace(emoji, text_rep)
    # Generic emoji removal for remaining unicode emojis outside map
    text = re.sub(r'[\U00010000-\U0010ffff]', ' ', text)
    return text


def expand_contractions(text: str) -> str:
    """Expands common English contractions to preserve sentiment context."""
    contractions = {
        r"can\'t": "cannot",
        r"won\'t": "will not",
        r"n\'t": " not",
        r"\'re": " are",
        r"\'s": " is",
        r"\'d": " would",
        r"\'ll": " will",
        r"\'t": " not",
        r"\'ve": " have",
        r"\'m": " am"
    }
    for pattern, repl in contractions.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def preprocess_text(text: str) -> str:
    """
    Main NLP preprocessing pipeline for a single text string.
    Steps:
    1. Handle non-string / NaN
    2. Replace emojis with text equivalents
    3. Expand contractions
    4. Lowercasing
    5. Remove URLs
    6. Remove @mentions
    7. Clean hashtags (preserve word: #awesome -> awesome)
    8. Remove digits/numbers
    9. Remove punctuation
    10. Remove extra whitespace
    11. Tokenize & Filter Stopwords (preserving negations)
    12. Lemmatization
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Emoji handling
    text = replace_emojis(text)

    # 2. Contraction expansion
    text = expand_contractions(text)

    # 3. Lowercase
    text = text.lower()

    # 4. URL removal
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 5. @mention removal
    text = re.sub(r'@\w+', '', text)

    # 6. Hashtag cleaning (#awesome -> awesome)
    text = re.sub(r'#(\w+)', r'\1', text)

    # 7. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 8. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 9. Extra whitespace removal
    text = re.sub(r'\s+', ' ', text).strip()

    # 10. Tokenization
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # 11. Stopword removal with negation preservation & Lemmatization
    cleaned_tokens = []
    for token in tokens:
        if token not in CUSTOM_STOPWORDS and len(token) > 1:
            if lemmatizer:
                try:
                    token = lemmatizer.lemmatize(token)
                except Exception:
                    pass
            cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)


def preprocess_text_step_by_step(text: str) -> dict:
    """
    Returns step-by-step intermediate results for academic / UI demonstration.
    """
    if not isinstance(text, str):
        text = ""

    step1_raw = text
    step2_emoji = replace_emojis(step1_raw)
    step3_contractions = expand_contractions(step2_emoji)
    step4_lowercase = step3_contractions.lower()
    step5_urls_mentions = re.sub(r'https?://\S+|www\.\S+', '', step4_lowercase)
    step5_urls_mentions = re.sub(r'@\w+', '', step5_urls_mentions)
    step6_hashtags = re.sub(r'#(\w+)', r'\1', step5_urls_mentions)
    step7_punctuation = step6_hashtags.translate(str.maketrans('', '', string.punctuation))
    step7_punctuation = re.sub(r'\d+', '', step7_punctuation)
    step8_whitespace = re.sub(r'\s+', ' ', step7_punctuation).strip()

    try:
        step9_tokens = word_tokenize(step8_whitespace)
    except Exception:
        step9_tokens = step8_whitespace.split()

    step10_stopwords_preserved = [t for t in step9_tokens if t not in CUSTOM_STOPWORDS and len(t) > 1]
    
    step11_lemmatized = []
    for t in step10_stopwords_preserved:
        if lemmatizer:
            try:
                t = lemmatizer.lemmatize(t)
            except Exception:
                pass
        step11_lemmatized.append(t)

    final_text = " ".join(step11_lemmatized)

    return {
        "1_original": step1_raw,
        "2_emoji_and_contractions": step3_contractions,
        "3_lowercased": step4_lowercase,
        "4_urls_mentions_removed": step5_urls_mentions,
        "5_hashtags_cleaned": step6_hashtags,
        "6_punctuation_digits_removed": step7_punctuation,
        "7_whitespace_cleaned": step8_whitespace,
        "8_tokens": step9_tokens,
        "9_stopwords_filtered": step10_stopwords_preserved,
        "10_lemmatized_final": final_text
    }
