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

async def get_document_by_id(db: AsyncSession, id_doc: int, id_dc: int):
    result = await db.execute(
        select(Document).where(Document.id == id_doc, Document.id_dc == id_dc)
    )
    return result.scalar_one_or_none()