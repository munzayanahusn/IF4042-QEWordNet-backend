import math
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_tokens(tokens, use_stop=False, use_stem=False):
    if use_stop:
        tokens = [t for t in tokens if t not in stop_words]
    if use_stem:
        tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def compute_tf_log(tf: int) -> float:
    return 1 + math.log2(tf) if tf > 0 else 0

def compute_tf_binary(tf: int) -> float:
    return 1 if tf > 0 else 0

def compute_tf_augmented(tf: int, max_tf: int) -> float:
    return 0.5 + 0.5 * (tf / max_tf) if max_tf > 0 else 0

def compute_idf(df: int, N: int) -> float:
    return math.log2(N / df) if df > 0 else 0