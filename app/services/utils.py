from typing import Dict, List, Tuple

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
        
        sorted_retrieval = sorted(
            retrieval_comparison, 
            key=lambda x: x['expanded_rank'] if x['expanded_rank'] is not None else float('inf')
        )
        
        for i, result in enumerate(sorted_retrieval, 1):
            initial_rank = result['initial_rank'] if result['initial_rank'] else 'N/A'
            expanded_rank = result['expanded_rank'] if result['expanded_rank'] else 'N/A'
            
            output_lines.append(
                f"{i}. doc_id: {result['doc_id']}, "
                f"initial_score: {result['initial_score']:.3f}, "
                f"expanded_score: {result['expanded_score']:.3f}, "
                f"initial_rank: {initial_rank}, "
                f"expanded_rank: {expanded_rank}"
            )
    
    return "\n".join(output_lines)