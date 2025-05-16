from pydantic import BaseModel

class DocumentCreate(BaseModel):
    id_doc: int
    id_dc: int
    title: str
    author: str
    content: str

class DocumentOut(DocumentCreate):
    id: int

    class Config:
        orm_mode = True
