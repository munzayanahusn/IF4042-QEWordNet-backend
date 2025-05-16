from fastapi import FastAPI
from app.api.endpoints import user

app = FastAPI()
app.include_router(user.router)
