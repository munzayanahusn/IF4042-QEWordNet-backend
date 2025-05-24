import time

from fastapi import APIRouter, Query, HTTPException, Depends
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import AsyncSessionLocal
from app.services.search_engine import search_query

router = APIRouter(tags=["Search"])

VALID_SYNSET_TYPES = {
    "lemmas",
    "hyponyms",
    "hypernyms",
    "also_sees",
    "similar_tos",
    "verb_groups"
}

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.get("/search/")
async def search(
    dc_id: int,
    query: str,
    synset: List[str] = Query(None),
    stem: bool = False,
    stopword: bool = False,
    query_tf: str = Query("raw", enum=["raw", "log", "augmented", "binary"]),
    query_idf: bool = False,
    query_norm: bool = False,
    doc_tf: str = Query("raw", enum=["raw", "log", "augmented", "binary"]),
    doc_idf: bool = False,
    doc_norm: bool = False,
    db: AsyncSession = Depends(get_db),
):
    try:
        # Validate input parameters
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        if not isinstance(dc_id, int) or dc_id <= 0:
            raise HTTPException(status_code=400, detail="Document collection ID must be a positive integer")
        if not isinstance(stem, bool):
            raise HTTPException(status_code=400, detail="Stem must be a boolean value")
        if not isinstance(stopword, bool):
            raise HTTPException(status_code=400, detail="Stopword must be a boolean value")
        if query_tf not in ["raw", "log", "augmented", "binary"]:
            raise HTTPException(status_code=400, detail="Invalid query TF type")
        if doc_tf not in ["raw", "log", "augmented", "binary"]:
            raise HTTPException(status_code=400, detail="Invalid document TF type")
        if not isinstance(query_idf, bool):
            raise HTTPException(status_code=400, detail="Query IDF must be a boolean value")
        if not isinstance(query_norm, bool):
            raise HTTPException(status_code=400, detail="Query normalization must be a boolean value")
        if not isinstance(doc_idf, bool):
            raise HTTPException(status_code=400, detail="Document IDF must be a boolean value")
        if not isinstance(doc_norm, bool):
            raise HTTPException(status_code=400, detail="Document normalization must be a boolean value")
        if not isinstance(synset, list):
            raise HTTPException(status_code=400, detail="Synset must be a list of strings")

        for s in synset:
            if s not in VALID_SYNSET_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid synset type: '{s}'. Allowed types: {', '.join(sorted(VALID_SYNSET_TYPES))}"
                )
            
        results = await search_query(
            db=db,
            dc_id=dc_id,
            query=query,
            synset=synset,
            stem=stem,
            stopword=stopword,
            query_tf=query_tf,
            query_idf=query_idf,
            query_norm=query_norm,
            doc_tf=doc_tf,
            doc_idf=doc_idf,
            doc_norm=doc_norm
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
