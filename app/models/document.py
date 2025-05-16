from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.base import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    id_doc = Column(Integer, nullable=False)
    id_dc = Column(Integer, ForeignKey("document_collections.id"), nullable=False)

    title = Column(String, nullable=True)
    author = Column(String, nullable=True)
    content = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint('id_doc', 'id_dc', name='uq_document_id_dc'),
    )

    collection = relationship("DocumentCollection", back_populates="documents")