from typing import List, Dict, Set, Optional
from pydantic import BaseModel

class QueryInput(BaseModel):
    dc_id: int
    query_id: int
    query_text: str
    relevant_docs: Set[int]
    settings: Dict
    
class BatchQuerySettings(BaseModel):
    query_tf: str = 'raw'
    query_idf: bool = False
    query_norm: bool = False
    doc_tf: str = 'raw'
    doc_idf: bool = False
    doc_norm: bool = False
    stem: bool = False
    stopword: bool = False
    synsets: List[str] = []