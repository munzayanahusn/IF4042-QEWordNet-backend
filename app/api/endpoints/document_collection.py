from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.schemas.document_collection import DocumentCollectionOut
from app.schemas.inverted import InvertedEntry
from app.services.search_engine import read_inverted_file_by_dc
from app.crud.document_collection import get_all_document_collections, get_document_collection_by_id

import os
import glob
import csv

router = APIRouter(tags=["Document Collection"])

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.get("/dc/all/", response_model=list[DocumentCollectionOut])
async def get_all_document_collections_route(
    db: AsyncSession = Depends(get_db)
):
    return await get_all_document_collections(db)

@router.get("/inverted/", response_model=list[InvertedEntry])
async def read_inverted_by_id(
    id_dc: int = Query(None),
    id_doc: int = Query(None),
    stem: bool = Query(False),
    stopword: bool = Query(False),
    db: AsyncSession = Depends(get_db)
):  
    if id_dc is None:
        raise HTTPException(status_code=400, detail="Document collection ID is required")
    if id_doc is None:
        raise HTTPException(status_code=400, detail="Document ID is required")
    
    dc = await get_document_collection_by_id(db, id_dc)
    if not dc:
        raise HTTPException(status_code=404, detail=f"Document collection with ID {id_dc} not found")

    # Read inverted file
    try:
        data = read_inverted_file_by_dc(dc, stem, stopword)
        filtered_data = [item for item in data if item["doc_id"] == id_doc]

        return [InvertedEntry(**row) for row in filtered_data]
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))