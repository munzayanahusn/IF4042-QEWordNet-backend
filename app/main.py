from fastapi import FastAPI
from app.api.endpoints import user, upload

app = FastAPI()
app.include_router(user.router)
app.include_router(upload.router)