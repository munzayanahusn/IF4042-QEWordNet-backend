import math
import nltk

from typing import Dict, List, Set, Optional, Tuple
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

def calculate_average_precision(ranked_results: List[Dict], relevant_docs: Set[int]) -> float:
    """Calculate Average Precision (AP)"""

    print("[DEBUG] AVERAGE PRECISION")
    print(ranked_results)
    print(relevant_docs)
    
    if not relevant_docs:
        return 0.0
    
    precisions = []
    relevant_count = 0
    
    for i, result in enumerate(ranked_results):
        if result['doc_id'] in relevant_docs:
            relevant_count += 1
            precision = relevant_count / (i + 1)
            precisions.append(precision)
    
    return sum(precisions) / len(relevant_docs) if precisions else 0.0

def calculate_mean_average_precision(all_results: Dict[int, List[Dict]], relevant_docs: Dict[int, Set[int]]) -> float:
    """Calculate Mean Average Precision (MAP)"""
    average_precisions = []
    
    for query_id, results in all_results.items():
        if query_id in relevant_docs and len(relevant_docs[query_id]) > 0:
            ap = calculate_average_precision(results, relevant_docs[query_id])
            average_precisions.append(ap)
    
    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0