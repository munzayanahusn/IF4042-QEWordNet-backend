import nltk

from fastapi import FastAPI
from nltk.data import find
from nltk import download

try:
    find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

try:
    find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

from app.api.endpoints import user, upload, document_collection, search

app = FastAPI()
app.include_router(user.router)
app.include_router(upload.router)
app.include_router(document_collection.router)
app.include_router(search.router)