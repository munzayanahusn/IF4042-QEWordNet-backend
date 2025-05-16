import csv
import os
import math
import glob
from collections import defaultdict, Counter
from typing import Dict, List
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.calculation import preprocess_tokens, compute_tf_log, compute_tf_augmented, compute_tf_binary, compute_idf
from app.crud.document_collection import get_document_collection_by_id
from app.schemas.inverted import InvertedEntry


def load_inverted_index(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            {
                "term": row["term"],
                "doc_id": int(row["doc_id"]),
                "tf_raw": int(row["tf_raw"]),
                "tf_log": float(row["tf_log"]),
                "tf_binary": int(row["tf_binary"]),
                "tf_augmented": float(row["tf_augmented"]),
                "idf": float(row["idf"]),
            }
            for row in reader
        ]

def read_inverted_file_by_dc(dc, stem: bool, stopword: bool):
    if stem and stopword:
        inverted_file = "*_stem_stop.csv"
    elif stem:
        inverted_file = "*_stem.csv"
    elif stopword:
        inverted_file = "*_stop.csv"
    else:
        inverted_file = "*_normal.csv"

    candidate_files = glob.glob(os.path.join(dc.inverted_path, inverted_file))
    if not candidate_files:
        raise FileNotFoundError(f"No matching inverted file for stemming={stem}, stopword={stopword}")

    return load_inverted_index(candidate_files[0])

def get_tf_weight(tf_type: str, tf_raw: int, doc_tokens=None) -> float:
    if tf_type == "raw":
        return tf_raw
    elif tf_type == "log":
        return compute_tf_log(tf_raw)
    elif tf_type == "augmented" and doc_tokens is not None:
        max_tf = max(Counter(doc_tokens).values())
        return compute_tf_augmented(tf_raw, max_tf)
    elif tf_type == "binary":
        return compute_tf_binary(tf_raw)
    else:
        raise ValueError(f"Unknown TF type: {tf_type}")

def search_internal(
    dc,
    query: str,
    stem: bool,
    stopword: bool,
    query_tf: str,
    query_idf: bool,
    query_norm: bool,
    doc_tf: str,
    doc_idf: bool,
    doc_norm: bool
):
    inverted_data = read_inverted_file_by_dc(dc, stem, stopword)
    if not inverted_data:
        raise Exception("No inverted data found")
    
    doc_id_map = {}
    for doc in dc.documents:
        doc_id_map[doc.id_doc] = {
            "title": doc.title,
            "author": doc.author,
            "content": doc.content
        }

    doc_vectors = defaultdict(lambda: defaultdict(float))
    idf_lookup = {}
    for entry in inverted_data:
        term = entry["term"]
        doc_id = entry["doc_id"]
        tf_weight = get_tf_weight(doc_tf, entry["tf_raw"])
        if doc_idf:
            tf_weight *= entry["idf"]
        doc_vectors[doc_id][term] = tf_weight
        idf_lookup[term] = entry["idf"]

    tokens = word_tokenize(query.lower())
    tokens = [t for t in tokens if t.isalnum()]
    tokens = preprocess_tokens(tokens, stopword, stem)
    query_counts = Counter(tokens)

    query_vector = {}
    for term, tf in query_counts.items():
        tf_weight = get_tf_weight(query_tf, tf, tokens)
        if query_idf:
            tf_weight *= idf_lookup.get(term, 0)
        query_vector[term] = tf_weight

    results = []
    for doc_id, vector in doc_vectors.items():
        dot_product = sum(query_vector[t] * vector.get(t, 0.0) for t in query_vector)

        if query_norm:
            q_norm = math.sqrt(sum(v**2 for v in query_vector.values()))
            dot_product /= q_norm if q_norm != 0 else 1

        if doc_norm:
            d_norm = math.sqrt(sum(v**2 for v in vector.values()))
            dot_product /= d_norm if d_norm != 0 else 1
        
        doc_info = doc_id_map.get(doc_id, {
            "title": "Unknown", 
            "author": "Unknown", 
            "content": "Unknown"
        })

        results.append({
            "doc_id": doc_id,
            "score": dot_product,
            "doc_title": doc_info["title"],
            "doc_author": doc_info["author"],
            "doc_content": doc_info["content"]
        })

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    return {
        "ranked_results": [
            {
                "doc_id": r["doc_id"],
                "doc_title": r["doc_title"],
                "doc_author": r["doc_author"],
                "doc_content": r["doc_content"],
                "score": r["score"],
                "rank": i + 1
            }
            for i, r in enumerate(ranked)
        ],
        "query_vector": query_vector
    }

def get_wordnet_synonyms(word: str) -> List[str]:
    return list({lemma.name().replace("_", " ") for syn in wordnet.synsets(word) for lemma in syn.lemmas()})

def get_all_synonyms(query: str) -> List[str]:
    tokens = word_tokenize(query.lower())
    all_synonyms = set()
    for token in tokens:
        all_synonyms.update(get_wordnet_synonyms(token))
    return list(all_synonyms)

async def search_query(
    db: AsyncSession,
    dc_id: int,
    query: str,
    stem: bool,
    stopword: bool,
    query_tf: str,
    query_idf: bool,
    query_norm: bool,
    doc_tf: str,
    doc_idf: bool,
    doc_norm: bool
):
    dc = await get_document_collection_by_id(db, dc_id)
    if not dc:
        raise Exception("Document collection not found")

    # Search initial query
    initial = search_internal(
        dc,
        query,
        stem,
        stopword,
        query_tf,
        query_idf,
        query_norm,
        doc_tf,
        doc_idf,
        doc_norm
    )

    # Expand query using WordNet
    expanded_terms = set(word_tokenize(query.lower())).union(get_all_synonyms(query))
    expanded_query = " ".join(expanded_terms)

    # Search expanded query
    expanded = search_internal(
        dc,
        expanded_query,
        stem,
        stopword,
        query_tf,
        query_idf,
        query_norm,
        doc_tf,
        doc_idf,
        doc_norm
    )

    return {
        "initial_query": query,
        "initial_query_vector": initial["query_vector"],
        "initial_results": initial["ranked_results"],
        "expanded_query": expanded_query,
        "expanded_query_vector": expanded["query_vector"],
        "expanded_results": expanded["ranked_results"]
    }