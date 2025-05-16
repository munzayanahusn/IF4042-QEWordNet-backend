import os
import csv

from collections import defaultdict, Counter
from typing import List, Tuple
from nltk.tokenize import word_tokenize

from app.services.calculation import compute_tf_log, compute_tf_binary, compute_tf_augmented, compute_idf, preprocess_tokens

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
                idf = compute_idf(df, N)

            for doc_id, tf in doc_freqs.items():
                tf_binary = compute_tf_binary(tf)
                tf_log = compute_tf_log(tf)

                max_tf = max(Counter(doc_tokens[doc_id]).values())
                tf_aug = compute_tf_augmented(tf, max_tf)

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
