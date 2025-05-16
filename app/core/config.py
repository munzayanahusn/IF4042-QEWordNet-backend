from dotenv import load_dotenv
import os

load_dotenv()  # ✅ Load .env from root

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

# Check if any variable is None
if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    raise Exception("Database environment variables not loaded. Check your .env file.")

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
