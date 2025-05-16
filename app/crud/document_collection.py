from sqlalchemy.ext.asyncio import AsyncSession
from app.models.document_collection import DocumentCollection
from app.schemas.document_collection import DocumentCollectionCreate

async def create_document_collection(
    db: AsyncSession,
    dc: DocumentCollectionCreate
) -> DocumentCollection:
    db_dc = DocumentCollection(**dc.dict())
    db.add(db_dc)
    await db.commit()
    await db.refresh(db_dc)
    return db_dc
