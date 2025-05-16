from fastapi import APIRouter, Query, HTTPException, Depends
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import AsyncSessionLocal
from app.services.search_engine import search_query

router = APIRouter(tags=["Search"])

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.get("/search/")
async def search(
    dc_id: int,
    query: str,
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
        results = await search_query(
            db=db,
            dc_id=dc_id,
            query=query,
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
