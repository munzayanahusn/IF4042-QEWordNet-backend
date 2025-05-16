from fastapi import FastAPI
from app.api.endpoints import user, upload, document_collection, search

app = FastAPI()
app.include_router(user.router)
app.include_router(upload.router)
app.include_router(document_collection.router)
app.include_router(search.router)