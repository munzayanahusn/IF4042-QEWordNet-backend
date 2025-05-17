import nltk

from fastapi import FastAPI
from nltk.data import find
from nltk import download

try:
    find("tokenizers/punkt")
except LookupError:
    download("punkt")

try:
    find("corpora/stopwords")
except LookupError:
    download("stopwords")

from app.api.endpoints import user, upload, document_collection, search

app = FastAPI()
app.include_router(user.router)
app.include_router(upload.router)
app.include_router(document_collection.router)
app.include_router(search.router)