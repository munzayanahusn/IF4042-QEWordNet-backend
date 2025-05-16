from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import AsyncSessionLocal
from app.schemas.document_collection import DocumentCollectionCreate
from app.schemas.document import DocumentCreate
from app.crud.document_collection import create_document_collection
from app.crud.document import bulk_create_documents
from app.services.parser import parse_and_generate
import os, shutil
import uuid

router = APIRouter()

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    # Save file
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    doc_path = os.path.join("storage", "docs", filename)
    with open(doc_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Parse & generate inverted index
    parsed_docs, inverted_path = parse_and_generate(doc_path)

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
