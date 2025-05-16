from pydantic import BaseModel

class DocumentCollectionCreate(BaseModel):
    dc_path: str
    inverted_path: str

class DocumentCollectionOut(DocumentCollectionCreate):
    id: int

    class Config:
        orm_mode = True
