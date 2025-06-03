# STBI WordNet
STBI WordNet is a web-based application designed to experiment with improving text document search relevance through WordNet-based query expansion. The application enables users to perform document searches using input queries alongside their expanded forms, and provides detailed metrics on query expansion effectiveness.

## Features

- **Document Management:** Upload and manage collections of text documents. View inverted indexes with various TF-IDF weighting schemes (raw TF, log TF, binary TF, augmented TF, IDF). Supports viewing original, stemmed, stop-word removed, or combined text processing variants.  
- **Interactive Search:** Perform searches with original and expanded queries. Display search results ranked by cosine similarity. Compare rankings, view expanded query terms, and analyze term weight differences. Configure weighting methods and text preprocessing options.  
- **Batch Search:** Run batch queries with relevance judgments. Automatically expand queries and calculate precision metrics such as average precision and mean average precision. Compare effectiveness between original and expanded query sets.  
- **Detailed Metrics:** Visualize term weighting and query expansion impact to aid in relevance evaluation.

## Program Requirements

- Python 
- PostgreSQL 
- Libraries and frameworks:  
  - FastAPI
  - SQLAlchemy
  - NLTK
  - Pydantic  
- Additional dependencies as listed in `requirements.txt`

## How to Run the Program

1. **Backend Setup:**  
   - Install Python dependencies:  
     ```bash
     pip install -r requirements.txt
     ```
     ```bash
     pip install Asyncpg FastAPI multipart-python alchemy dotenv uvicorn alembic psycopg2 aiofiles
     ```
   - Configure PostgreSQL database
   - Update .env settings  
   - Initialize the database schema:
     ```bash
     alembic upgrade head
     ```
   - Run the FastAPI backend server:  
     ```bash
     python run.py
     ```


2. **Access the Documentation:**  
   - Open a web browser
   - Navigate to ` http://127.0.0.1:8000/docs` to access the API Documentation.

## Authors
- [13521047] Muhammad Equilibrie Fajria  
- [13521053] Athif Nirwasito  
- [13521077] Husnia Munzayana  
- [13521115] Shelma Salsabila  
- [13521125] Asyifa Nurul Shafira  
