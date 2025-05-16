import os
import uuid
import math
import csv
from collections import defaultdict, Counter
from typing import List, Tuple
import nltk

from nltk.tokenize import word_tokenize
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

def parse_and_generate(file_path: str) -> Tuple[List[dict], str]:
    # Parse raw document
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    documents = []
    current = {"id_doc": None, "title": "", "author": "", "content": ""}
    section = None

    for line in lines:
        line = line.rstrip()
        if line.startswith(".I"):
            if current["id_doc"] is not None:
                documents.append(current)
                current = {"id_doc": None, "title": "", "author": "", "content": ""}
            current["id_doc"] = int(line.split()[1])
        elif line.startswith(".T"):
            section = "title"
        elif line.startswith(".A"):
            section = "author"
        elif line.startswith(".W"):
            section = "content"
        elif line.startswith(".X"):
            section = None
        elif section:
            current[section] += " " + line.strip()

    if current["id_doc"] is not None:
        documents.append(current)

    # Output directory
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join("storage", "inverted", base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Process inverted file
    options = [
        (False, False, "normal"),
        (True,  False, "stop"),
        (False, True,  "stem"),
        (True,  True,  "stem_stop"),
    ]

    for use_stop, use_stem, suffix in options:
        doc_tokens = {}
        term_doc_freqs = defaultdict(lambda: defaultdict(int))

        for doc in documents:
            content = f"{doc['title']} {doc['content']}"
            tokens = word_tokenize(content.lower())
            tokens = [t for t in tokens if t.isalnum()]
            tokens = preprocess_tokens(tokens, use_stop, use_stem)

            doc_tokens[doc["id_doc"]] = tokens
            counts = Counter(tokens)
            for term, count in counts.items():
                term_doc_freqs[term][doc["id_doc"]] = count

        N = len(documents)
        inverted_data = []

        for term, doc_freqs in term_doc_freqs.items():
            df = len(doc_freqs)

            if df == 0:
                idf = 0
            else:
                idf = math.log2(N / df)

            for doc_id, tf in doc_freqs.items():
                tf_binary = 1
                tf_log = 1 + math.log2(tf)
                tf_aug = 0.5 + 0.5 * (tf / max(Counter(doc_tokens[doc_id]).values()))

                inverted_data.append({
                    "term": term,
                    "doc_id": doc_id,
                    "tf_raw": tf,
                    "tf_log": tf_log,
                    "tf_binary": tf_binary,
                    "tf_augmented": tf_aug,
                    "idf": idf
                })
        
        sorted_data = sorted(inverted_data, key=lambda x: x["term"])

        output_path = os.path.join(output_dir, f"{base_name}_{suffix}.csv")
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "term", 
                "doc_id", 
                "tf_raw", 
                "tf_log", 
                "tf_binary", 
                "tf_augmented", 
                "idf"
            ])
            writer.writeheader()
            writer.writerows(sorted_data)

    # Return documents and folder path
    return documents, output_dir
