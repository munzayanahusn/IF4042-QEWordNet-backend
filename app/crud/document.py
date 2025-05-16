from sqlalchemy.ext.asyncio import AsyncSession
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
