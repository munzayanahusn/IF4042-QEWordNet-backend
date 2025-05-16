from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List

from app.models.document_collection import DocumentCollection
from app.schemas.document_collection import DocumentCollectionCreate, DocumentCollectionOut

async def create_document_collection(
    db: AsyncSession,
    dc: DocumentCollectionCreate
) -> DocumentCollection:
    db_dc = DocumentCollection(**dc.dict())
    db.add(db_dc)
    await db.commit()
    await db.refresh(db_dc)
    return db_dc

async def get_all_document_collections(db: AsyncSession):
    result = await db.execute(select(DocumentCollection))
    return result.scalars().all()

async def get_document_collection_by_id(db: AsyncSession, id_dc: int):
    result = await db.execute(
        select(DocumentCollection).where(DocumentCollection.id == id_dc)
    )
    return result.scalar_one_or_none()