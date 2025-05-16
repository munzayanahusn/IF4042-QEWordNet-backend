from pydantic import BaseModel

class DocumentCollectionCreate(BaseModel):
    dc_path: str
    inverted_path: str

class DocumentCollectionOut(DocumentCollectionCreate):
    id: int
    dc_path: str
    inverted_path: str

    class Config:
        orm_mode = True
