from sqlalchemy import Column, Integer, String
from app.db.base import Base

class DocumentCollection(Base):
    __tablename__ = "document_collections"

    id = Column(Integer, primary_key=True, index=True)
    dc_path = Column(String, nullable=False)
    inverted_path = Column(String, nullable=False)