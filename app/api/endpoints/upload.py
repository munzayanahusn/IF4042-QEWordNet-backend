from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import AsyncSessionLocal
from app.schemas.document_collection import DocumentCollectionCreate
from app.schemas.document import DocumentCreate
from app.crud.document_collection import create_document_collection
from app.crud.document import bulk_create_documents
from app.services.parser import parse_and_generate
import os
import uuid
import aiofiles

router = APIRouter(tags=["Document Collection"])

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    # Save file asynchronously
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    doc_path = os.path.join("storage", "docs", filename)

    async with aiofiles.open(doc_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    # Parse & generate inverted index
    parsed_docs, inverted_path = await parse_and_generate(doc_path)

    # Save document_collection
    dc_data = DocumentCollectionCreate(dc_path=doc_path, inverted_path=inverted_path)
    dc = await create_document_collection(db, dc_data)

    # Save documents linked to collection
    doc_records = [DocumentCreate(
        id_doc=d["id_doc"],
        id_dc=dc.id,
        title=d["title"],
        author=d["author"],
        content=d["content"]
    ) for d in parsed_docs]
    await bulk_create_documents(db, doc_records)

    return {"message": "Upload successful", "collection_id": dc.id}
