from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.db.session import AsyncSessionLocal
from app.schemas.document_collection import DocumentCollectionOut
from app.schemas.inverted import InvertedEntry
from app.crud.document_collection import get_all_document_collections, get_document_collection_by_id
from app.crud.document import bulk_create_documents
from app.services.parser import parse_and_generate

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

@router.get("/inverted/{id_dc}", response_model=list[InvertedEntry])
async def read_inverted_by_id(
    id_dc: int, 
    stem: bool = Query(False),
    stopword: bool = Query(False),
    db: AsyncSession = Depends(get_db)
):
    # Get document collection
    dc = await get_document_collection_by_id(db, id_dc)
    if not dc:
        raise HTTPException(status_code=404, detail="Document collection not found")

    # Get inverted file path
    inverted_folder = dc.inverted_path

    if stem and stopword:
        inverted_file = "*_stem_stop.csv"
    elif stem:
        inverted_file = "*_stem.csv"
    elif stopword:
        inverted_file = "*_stop.csv"
    else:
        inverted_file = "*_normal.csv"

    candidate_files = glob.glob(os.path.join(inverted_folder, inverted_file))
    if not candidate_files:
        raise HTTPException(status_code=404, detail=f"No matching inverted file for stemming={stem}, stopword={stopword}")

    inverted_path = candidate_files[0]
    if not os.path.exists(inverted_path):
        raise HTTPException(status_code=404, detail="Inverted file not found")

    # Read and return inverted CSV
    try:
        with open(inverted_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [
                InvertedEntry(
                    term=row["term"],
                    doc_id=int(row["doc_id"]),
                    tf_raw=int(row["tf_raw"]),
                    tf_log=float(row["tf_log"]),
                    tf_binary=int(row["tf_binary"]),
                    tf_augmented=float(row["tf_augmented"]),
                    idf=float(row["idf"]),
                )
                for row in reader
            ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read inverted file: {e}")