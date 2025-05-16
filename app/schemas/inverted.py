from pydantic import BaseModel

class InvertedEntry(BaseModel):
    term: str
    doc_id: int
    tf_raw: int
    tf_log: float
    tf_binary: int
    tf_augmented: float
    idf: float
