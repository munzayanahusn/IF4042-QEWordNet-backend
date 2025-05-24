import csv
import os
import math
import glob
import time
from collections import defaultdict, Counter
from typing import Dict, List
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.calculation import preprocess_tokens, compute_tf_log, compute_tf_augmented, compute_tf_binary, compute_idf, calculate_average_precision, calculate_mean_average_precision
from app.services.utils import create_retrieval_comparison, create_term_weights_comparison, generate_formatted_output
from app.crud.document_collection import get_document_collection_by_id
from app.crud.document import get_doc_id_by_dc
from app.schemas.inverted import InvertedEntry
from app.schemas.query import QueryInput

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

async def search_internal(
    db: AsyncSession,
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

    doc_vectors = defaultdict(lambda: defaultdict(float))
    idf_lookup = {}
    doc_id_lookup = {}

    for entry in inverted_data:
        dc_doc_id = entry["doc_id"]
        if dc_doc_id not in doc_id_lookup:
            doc_id_lookup[dc_doc_id] = await get_doc_id_by_dc(db, dc.id, dc_doc_id)

        term = entry["term"]
        if doc_tf == "raw":
            tf_weight = entry["tf_raw"]
        elif doc_tf == "log":
            tf_weight = entry["tf_log"]
        elif doc_tf == "augmented":
            tf_weight = entry["tf_augmented"]
        elif doc_tf == "binary":
            tf_weight = entry["tf_binary"]
        else:
            raise ValueError(f"Unknown TF type: {doc_tf}")
        
        if doc_idf:
            tf_weight *= entry["idf"]
        doc_vectors[dc_doc_id][term] = tf_weight
        idf_lookup[term] = entry["idf"]

    tokens = word_tokenize(query.lower())
    tokens = [t.replace("_", " ") for t in tokens if t.isalnum() or "_" in t]

    # print("[DEBUG] Tokens before preprocessing:", tokens)
    tokens = preprocess_tokens(tokens, stopword, stem)
    # print("[DEBUG] Tokens after preprocessing:", tokens)

    query_counts = Counter(tokens)

    query_vector = {}
    for term, tf in query_counts.items():
        tf_weight = get_tf_weight(query_tf, tf, tokens)
        if query_idf:
            tf_weight *= idf_lookup.get(term, 0)
        query_vector[term] = tf_weight

    results = []
    for dc_doc_id, vector in doc_vectors.items():
        dot_product = sum(query_vector[t] * vector.get(t, 0.0) for t in query_vector)

        if query_norm:
            q_norm = math.sqrt(sum(v**2 for v in query_vector.values()))
            dot_product /= q_norm if q_norm != 0 else 1

        if doc_norm:
            d_norm = math.sqrt(sum(v**2 for v in vector.values()))
            dot_product /= d_norm if d_norm != 0 else 1

        results.append({
            "doc_id": doc_id_lookup[dc_doc_id],
            "dc_doc_id": dc_doc_id,
            "score": dot_product
        })

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    return {
        "ranked_results": [
            {
                "doc_id": r["doc_id"],
                "dc_doc_id": r["dc_doc_id"], 
                "score": r["score"], 
                "rank": i + 1}
            for i, r in enumerate(ranked)
        ],
        "query_vector": query_vector
    }

def get_wordnet_expansions(word: str, types: List[str]) -> List[str]:
    expansions = set()
    for syn in wordnet.synsets(word):
        for t in types:
            if t == "lemmas":
                expansions.update(syn.lemma_names())
            elif t == "hyponyms":
                expansions.update(hypo.name().split('.')[0] for hypo in syn.hyponyms())
            elif t == "hypernyms":
                expansions.update(hyper.name().split('.')[0] for hyper in syn.hypernyms())
            elif t == "also_sees":
                expansions.update(also.name().split('.')[0] for also in syn.also_sees())
            elif t == "similar_tos":
                expansions.update(similar.name().split('.')[0] for similar in syn.similar_tos())
            elif t == "verb_groups":
                expansions.update(verb.name().split('.')[0] for verb in syn.verb_groups())
            else:
                raise ValueError(f"Unknown synset type: {t}")
            
    cleaned_expansions = set()
    for expansion in expansions:
        cleaned_word = expansion.lower()
        if cleaned_word != word.lower():
            cleaned_expansions.add(cleaned_word)
    
    return list(cleaned_expansions)

async def search_query(
    db: AsyncSession,
    dc_id: int,
    query: str,
    synset: List[str],
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

    start_time = time.time()
    initial = await search_internal(db, dc, query, stem, stopword, query_tf, query_idf, query_norm, doc_tf, doc_idf, doc_norm)

    tokens = word_tokenize(query.lower())
    tokens = [t.replace("_", " ") for t in tokens if t.isalnum() or "_" in t]

    expansion_set = set()
    for token in tokens:
        expansion_set.update(get_wordnet_expansions(token, synset))

    expansion_set -= set(tokens)
    expanded_query = " ".join(tokens + sorted(expansion_set))

    expanded = await search_internal(db, dc, expanded_query, stem, stopword, query_tf, query_idf, query_norm, doc_tf, doc_idf, doc_norm)

    expanded_query = expanded_query.replace('_', ' ')
    elapsed_time = time.time() - start_time
    print(f"[DEBUG] Search time: {elapsed_time:.2f} seconds")

    return {
        "initial_query": query,
        "initial_query_vector": initial["query_vector"],
        "initial_results": initial["ranked_results"],
        "expanded_query": expanded_query,
        "expanded_query_vector": expanded["query_vector"],
        "expanded_results": expanded["ranked_results"],
        "elapsed_time": elapsed_time,
    }

async def search_query_batch(
    db: AsyncSession, 
    dc_id: int,
    queries: List[QueryInput]
) -> Dict:
    """
    Batch search function that processes multiple queries
    
    Args:
        db: Database session
        dc_id: Document collection ID
        queries: List of QueryInput objects containing query details
        
    Returns:
        Dict containing MAP scores and detailed results
    """
    # Process each query
    query_results = []
    all_initial_results = {} 
    all_expanded_results = {} 
    all_relevant_docs = {}
    batch_start_time = time.time()
    
    for query_input_obj in queries:
        try:
            # Access QueryInput object attributes correctly
            query_id = query_input_obj.query_id

            # Perform search
            query_start_time = time.time()

            search_result = await search_query(
                db=db, 
                dc_id=dc_id,
                query=query_input_obj.query_text,
                synset=query_input_obj.settings.get('synsets', []),
                stem=query_input_obj.settings.get('stem', False),
                stopword=query_input_obj.settings.get('stopword', False),
                query_tf=query_input_obj.settings.get('query_tf', 'raw'),
                query_idf=query_input_obj.settings.get('query_idf', False),
                query_norm=query_input_obj.settings.get('query_norm', False),
                doc_tf=query_input_obj.settings.get('doc_tf', 'raw'),
                doc_idf=query_input_obj.settings.get('doc_idf', False),
                doc_norm=query_input_obj.settings.get('doc_norm', False)
            )

            query_elapsed_time = time.time() - query_start_time
            print(f"[DEBUG] Searching time {query_input_obj.query_id}: {query_elapsed_time} s")
            
            # Get results for MAP calculation
            initial_results = search_result['initial_results']
            expanded_results = search_result.get('expanded_results', [])
            relevant_docs = query_input_obj.relevant_docs

            # Store for MAP calculation
            all_initial_results[query_id] = initial_results
            all_expanded_results[query_id] = expanded_results
            all_relevant_docs[query_id] = relevant_docs
            
            # Calculate individual AP scores
            initial_ap = calculate_average_precision(initial_results, relevant_docs)
            expanded_ap = calculate_average_precision(expanded_results, relevant_docs) if expanded_results else 0
            
            # Create retrieval comparison data
            retrieval_comparison = create_retrieval_comparison(initial_results, expanded_results)
            
            # Get term weights comparison
            initial_query_vector = search_result.get('initial_query_vector', {})
            expanded_query_vector = search_result.get('expanded_query_vector', {})
            term_weights = create_term_weights_comparison(initial_query_vector, expanded_query_vector)

            # Get expanded query from search result
            expanded_query = search_result.get('expanded_query', '')
            
            query_result = {
                'query_id': f'q{query_id:03d}',
                'initial_query': query_input_obj.query_text,
                'expanded_query': expanded_query,
                'initial_ap': initial_ap,
                'expanded_ap': expanded_ap,
                'term_weights': term_weights,
                'retrieval_comparison': retrieval_comparison,
                'relevant_docs': relevant_docs,
                'elapsed_time': query_elapsed_time
            }
            
            query_results.append(query_result)

        except Exception as e:
            print(f"Error processing query {query_input_obj.query_id}: {str(e)}")
            continue
    
    batch_elapsed_time = time.time() - batch_start_time
    print(f"[DEBUG] Searching time BATCH: {batch_elapsed_time} s")

    # Calculate MAP scores using your existing function
    map_initial = calculate_mean_average_precision(all_initial_results, all_relevant_docs)
    map_expanded = calculate_mean_average_precision(all_expanded_results, all_relevant_docs)
    
    # Generate formatted output file
    output_content = generate_formatted_output(map_initial, map_expanded, query_results)
    
    return {
        'elapsed_time': batch_elapsed_time,
        'map_initial': map_initial,
        'map_expanded': map_expanded,
        'query_results': [{
            'query_id': qr['query_id'],
            'initial_query': qr['initial_query'],
            'expanded_query': qr['expanded_query'], 
            'initial_ap': qr['initial_ap'],
            'expanded_ap': qr['expanded_ap']
        } for qr in query_results],
        'download_content': output_content,
        'processed_queries': len(query_results)
    }