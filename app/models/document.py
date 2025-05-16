from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint
from app.db.base import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    id_doc = Column(Integer, primary_key=False)
    id_dc = Column(Integer, ForeignKey("document_collections.id"), primary_key=False)

    title = Column(String, nullable=True)
    author = Column(String, nullable=True)
    content = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint('id_doc', 'id_dc', name='uq_document_id_dc'),
    )