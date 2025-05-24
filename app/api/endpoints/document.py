from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.schemas.document import DocumentOut
from app.crud.document import get_document_by_id, get_document_by_dc

router = APIRouter(tags=["Document"])

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.get("/docs/{id}", response_model=DocumentOut)
async def fetch_document_by_id(
    id: int,
    db: AsyncSession = Depends(get_db)
):
    doc = await get_document_by_id(db, id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID {id} not found")
    return doc

@router.get("/docs/", response_model=DocumentOut)
async def fetch_document_by_dc(
    id_dc: int,
    id_doc: int,
    db: AsyncSession = Depends(get_db)
):
    doc = await get_document_by_dc(db, id_dc=id_dc, id_doc=id_doc)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID_dc {id_dc} and ID_doc {id_doc} not found")
    return doc