import os
import csv
import time
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from concurrent.futures import ThreadPoolExecutor
import asyncio

from app.services.calculation import compute_tf_log, compute_tf_binary, compute_tf_augmented, compute_idf, preprocess_tokens
from app.schemas.query import QueryInput

VALID_SYNSET_TYPES = {
    "lemmas",
    "hyponyms",
    "hypernyms",
    "also_sees",
    "similar_tos",
    "verb_groups"
}

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

def parse_queries_file(content: str) -> Dict[int, str]:
    """Parse queries file with .I, .T, .A, .W format"""
    queries = {}
    current_id = None
    current_field = None
    title_content = ''
    article_content = ''
    
    lines = content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('.I '):
            if current_id is not None:
                combined_query = (title_content + ' ' + article_content).strip()
                if combined_query:
                    queries[current_id] = combined_query
            
            # Start new query
            current_id = int(line[3:].strip())
            title_content = ""
            article_content = ""
            current_field = None
            
        elif line.startswith('.T'):
            current_field = 'title'

        elif line.startswith('.W'):
            current_field = 'article'

        elif current_field and current_id is not None:
            # Only collect title and article content
            if current_field == 'title':
                if title_content:
                    title_content += ' ' + line
                else:
                    title_content = line
            elif current_field == 'article':
                if article_content:
                    article_content += ' ' + line
                else:
                    article_content = line
    
    # Save last query
    if current_id is not None:
        combined_query = (title_content + ' ' + article_content).strip()
        if combined_query:
            queries[current_id] = combined_query
    
    return queries

def parse_relevance_file(content: str) -> Dict[int, Set[int]]:
    """Parse relevance file format: query_id doc_id 0 0"""
    relevance = defaultdict(set)
    
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) >= 2:
            query_id = int(parts[0])
            doc_id = int(parts[1])
            relevance[query_id].add(doc_id)
    
    return dict(relevance)

def parse_smart_notation(smart: str) -> Tuple[str, str, bool]:
    """Parse smart notation into query and doc settings"""
    parts = smart.split('.')
    if len(parts) != 2:
        raise ValueError(f"Invalid smart notation: {smart}")
    
    query_part, doc_part = parts
    
    def parse_part(part: str) -> Tuple[str, bool, bool]:
        if len(part) != 3:
            raise ValueError(f"Invalid smart part: {part}")
        
        # TF: l=log, n=natural/raw, a=augmented, b=binary
        tf_type = part[0]  
        # IDF: t=use idf, n=no idf
        idf_flag = part[1] == 't'  
        # Normalization: c=normalize, n=no normalize
        norm_flag = part[2] == 'c'
        
        tf_mapping = {'l': 'log', 'n': 'raw', 'a': 'augmented', 'b': 'binary'}
        if tf_type not in tf_mapping:
            raise ValueError(f"Invalid TF type: {tf_type}")
            
        return tf_mapping[tf_type], idf_flag, norm_flag
    
    query_tf, query_idf, query_norm = parse_part(query_part)
    doc_tf, doc_idf, doc_norm = parse_part(doc_part)
    
    return (query_tf, query_idf, query_norm, doc_tf, doc_idf, doc_norm)

def parse_settings_file(content: str) -> Dict[int, Dict]:
    """Parse settings file"""
    settings = {}
    current_query_id = None
    current_setting = {}
    
    lines = content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('.I '):
            # Save previous setting if exists
            if current_query_id is not None:
                settings[current_query_id] = current_setting.copy()
            
            # Start new setting
            query_id_str = line[3:].strip()
            current_query_id = 0 if query_id_str == '0' else int(query_id_str)
            current_setting = {'synsets': []}
            
        elif line.startswith('.S '):
            smart_notation = line[3:].strip()
            try:
                query_tf, query_idf, query_norm, doc_tf, doc_idf, doc_norm = parse_smart_notation(smart_notation)
                current_setting.update({
                    'query_tf': query_tf,
                    'query_idf': query_idf,
                    'query_norm': query_norm,
                    'doc_tf': doc_tf,
                    'doc_idf': doc_idf,
                    'doc_norm': doc_norm
                })
            except ValueError as e:
                raise ValueError(f"Error parsing smart notation: {e}")
                
        elif line.startswith('.SM '):
            current_setting['stem'] = line[4:].strip().lower() == 'true'
            
        elif line.startswith('.SW '):
            current_setting['stopword'] = line[4:].strip().lower() == 'true'
            
        elif line.startswith('.SY '):
            # Synsets start here, following lines are synset types
            pass
        elif current_setting is not None and line in VALID_SYNSET_TYPES:
            current_setting['synsets'].append(line)
    
    # Save last setting
    if current_query_id is not None:
        settings[current_query_id] = current_setting
    
    return settings

def query_input(
    dc_id: int,
    parsed_queries: Dict[int, str],
    parsed_relevance: Dict[int, Set[int]],
    parsed_settings: Dict[int, Dict]
) -> List[QueryInput]:
    """
    Convert parsed inputs into structured query input list
    
    Args:
        dc_id: Document collection ID
        parsed_queries: Dict mapping query_id to query_text
        parsed_relevance: Dict mapping query_id to set of relevant doc_ids
        parsed_settings: Dict mapping query_id to settings dict
        
    Returns:
        List of QueryInput objects ready for batch processing
    """
    
    # Get default settings (query ID 0)
    default_settings = parsed_settings.get(0, {
        'query_tf': 'raw',
        'query_idf': False,
        'query_norm': False,
        'doc_tf': 'raw',
        'doc_idf': False,
        'doc_norm': False,
        'stem': False,
        'stopword': False,
        'synsets': []
    })
    
    queries = []
    
    for query_id, query_text in parsed_queries.items():
        if not query_text.strip():
            continue
            
        query_settings = parsed_settings.get(query_id, default_settings)
        relevant_docs = parsed_relevance.get(query_id, set())
        
        query_input_obj = QueryInput(
            dc_id=dc_id,
            query_id=query_id,
            query_text=query_text,
            relevant_docs=relevant_docs,
            settings=query_settings
        )
        
        queries.append(query_input_obj)
    
    return queries

def create_term_weights_comparison(
    initial_vector: Dict[str, float], 
    expanded_vector: Dict[str, float]
) -> Dict[str, Tuple[float, float]]:
    """Create term weights comparison between initial and expanded queries"""
    all_terms = set(initial_vector.keys()) | set(expanded_vector.keys())
    
    term_weights = {}
    for term in all_terms:
        initial_weight = initial_vector.get(term, 0.0)
        expanded_weight = expanded_vector.get(term, 0.0)
        term_weights[term] = (initial_weight, expanded_weight)
    
    return term_weights

def create_retrieval_comparison(
    initial_results: List[Dict], 
    expanded_results: List[Dict]
) -> List[Dict]:
    """Create retrieval comparison showing rank changes between initial and expanded results"""
    # Create lookup for expanded results
    expanded_lookup = {result['doc_id']: result for result in expanded_results} if expanded_results else {}
    
    # Get all unique doc_ids from both results
    all_doc_ids = set()
    initial_lookup = {result['doc_id']: result for result in initial_results}
    all_doc_ids.update(initial_lookup.keys())
    all_doc_ids.update(expanded_lookup.keys())
    
    comparison = []
    for doc_id in all_doc_ids:
        initial_result = initial_lookup.get(doc_id)
        expanded_result = expanded_lookup.get(doc_id)
        
        comparison_item = {
            'doc_id': f'D{doc_id}',
            'initial_score': initial_result['score'] if initial_result else 0.0,
            'expanded_score': expanded_result['score'] if expanded_result else 0.0,
            'initial_rank': initial_result['rank'] if initial_result else None,
            'expanded_rank': expanded_result['rank'] if expanded_result else None
        }
        comparison.append(comparison_item)
    
    # Sort by best rank (considering both initial and expanded)
    comparison.sort(key=lambda x: min(
        x['initial_rank'] if x['initial_rank'] else float('inf'),
        x['expanded_rank'] if x['expanded_rank'] else float('inf')
    ))
    
    return comparison

def generate_formatted_output(
    map_initial: float, 
    map_expanded: float, 
    query_results: List[Dict]
) -> str:
    """Generate formatted output content matching the required format"""
    output_lines = []
    
    # Header with MAP scores
    output_lines.append(f"MAP initial: {map_initial:.4f}")
    output_lines.append(f"MAP expanded: {map_expanded:.4f}")
    output_lines.append("")
    
    # Process each query
    for query_result in query_results:
        output_lines.append(f"Query ID: {query_result['query_id']}")
        output_lines.append("")

        # Query Text - Fixed the bug here
        output_lines.append(f"Initial Query: {query_result['initial_query']}")
        output_lines.append(f"Expanded Query: {query_result['expanded_query']}")
        output_lines.append("")
        
        # Term weights comparison
        output_lines.append("Term Weight: (term: initial, expanded)")
        term_weights = query_result['term_weights']
        for term, (initial_weight, expanded_weight) in sorted(term_weights.items()):
            output_lines.append(f"{term}: {initial_weight:.3f}, {expanded_weight:.3f}")
        output_lines.append("")
        
        # AP scores
        output_lines.append(f"AP Initial: {query_result['initial_ap']:.4f}")
        output_lines.append(f"AP Expanded: {query_result['expanded_ap']:.4f}")
        output_lines.append("")
        
        # Retrieval results
        output_lines.append("Retrieval Result:")
        retrieval_comparison = query_result['retrieval_comparison']
        
        for i, result in enumerate(retrieval_comparison[:10], 1):  # Show top 10 results
            initial_rank = result['initial_rank'] if result['initial_rank'] else 'N/A'
            expanded_rank = result['expanded_rank'] if result['expanded_rank'] else 'N/A'
            
            output_lines.append(
                f"{i}. doc_id: {result['doc_id']}, "
                f"initial_score: {result['initial_score']:.3f}, "
                f"expanded_score: {result['expanded_score']:.3f}, "
                f"initial_rank: {initial_rank}, "
                f"expanded_rank: {expanded_rank}"
            )
        
        output_lines.append("")
        output_lines.append("")
    
    return "\n".join(output_lines)