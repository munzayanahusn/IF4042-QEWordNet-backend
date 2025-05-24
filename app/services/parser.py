import os
import csv
import time
from collections import defaultdict, Counter
from typing import List, Tuple
from nltk.tokenize import word_tokenize
from concurrent.futures import ThreadPoolExecutor
import asyncio

from app.services.calculation import compute_tf_log, compute_tf_binary, compute_tf_augmented, compute_idf, preprocess_tokens

def process_single_doc(doc, use_stop, use_stem):
    content = f"{doc['title']} {doc['content']}"
    tokens = word_tokenize(content.lower())
    tokens = [t for t in tokens if t.isalnum()]
    tokens = preprocess_tokens(tokens, use_stop, use_stem)
    return doc["id_doc"], tokens, Counter(tokens)

async def parallel_process_docs(documents, use_stop, use_stem):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(executor, process_single_doc, doc, use_stop, use_stem)
            for doc in documents
        ]
        return await asyncio.gather(*tasks)

async def parse_and_generate(file_path: str) -> Tuple[List[dict], str]:
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

    # Processing options
    options = [
        (False, False, "normal"),
        (True,  False, "stop"),
        (False, True,  "stem"),
        (True,  True,  "stem_stop"),
    ]

    for use_stop, use_stem, suffix in options:
        start = time.time()

        results = await parallel_process_docs(documents, use_stop, use_stem)

        doc_tokens = {}
        term_doc_freqs = defaultdict(lambda: defaultdict(int))

        for doc_id, tokens, counts in results:
            doc_tokens[doc_id] = tokens
            for term, count in counts.items():
                term_doc_freqs[term][doc_id] = count

        # Precompute max_tf
        max_tfs = {
            doc_id: max(Counter(tokens).values()) if tokens else 1
            for doc_id, tokens in doc_tokens.items()
        }

        N = len(documents)
        inverted_data = []

        for term, doc_freqs in term_doc_freqs.items():
            df = len(doc_freqs)
            idf = compute_idf(df, N) if df > 0 else 0

            for doc_id, tf in doc_freqs.items():
                tf_log = compute_tf_log(tf)
                tf_binary = compute_tf_binary(tf)
                tf_aug = compute_tf_augmented(tf, max_tfs[doc_id])

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
                "term", "doc_id", "tf_raw", "tf_log", "tf_binary", "tf_augmented", "idf"
            ])
            writer.writeheader()
            writer.writerows(sorted_data)

        print(f"[{suffix}] Inverted index created in {time.time() - start:.2f}s")

    return documents, output_dir
