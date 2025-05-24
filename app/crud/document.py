from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.document import Document
from app.schemas.document import DocumentCreate
from typing import List

async def bulk_create_documents(
    db: AsyncSession,
    documents: List[DocumentCreate]
):
    objs = [Document(**d.dict()) for d in documents]
    db.add_all(objs)
    await db.commit()

async def get_document_by_dc(db: AsyncSession, id_dc: int, id_doc: int):
    result = await db.execute(
        select(Document).where(Document.id_dc == id_dc, Document.id_doc == id_doc)
    )
    return result.scalar_one_or_none()

async def get_document_by_id(db: AsyncSession, id: int):
    result = await db.execute(
        select(Document).where(Document.id == id)
    )
    return result.scalar_one_or_none()

async def get_document_id_by_dc(db: AsyncSession, id_dc: int):
    result = await db.execute(
        select(Document.id_doc)
        .where(Document.id_dc == id_dc)
        .order_by(Document.id_doc)
    )

    doc_ids = result.scalars().all()
    print("[DEBUG] len result:", len(doc_ids))
    return doc_ids

async def get_doc_id_by_dc(db: AsyncSession, id_dc: int, id_doc: int):
    result = await db.execute(
        select(Document.id)
        .where(Document.id_dc == id_dc, Document.id_doc == id_doc)
    )
    return result.scalar_one_or_none()