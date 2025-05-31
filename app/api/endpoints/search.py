import time
import json
from fastapi import APIRouter, Query, HTTPException, Depends, File, UploadFile, Form, Response
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AsyncSessionLocal
from app.services.search_engine import search_query, search_query_batch
from app.services.parser import (
    parse_queries_file,
    parse_relevance_file,
    parse_settings_file,
    query_input
)
from app.services.parser import VALID_SYNSET_TYPES

router = APIRouter(tags=["Search"])

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

@router.get("/search/")
async def search(
    dc_id: int,
    query: str,
    synset: List[str] = Query(None),
    stem: bool = False,
    stopword: bool = False,
    query_tf: str = Query("raw", enum=["raw", "log", "augmented", "binary"]),
    query_idf: bool = False,
    query_norm: bool = False,
    doc_tf: str = Query("raw", enum=["raw", "log", "augmented", "binary"]),
    doc_idf: bool = False,
    doc_norm: bool = False,
    db: AsyncSession = Depends(get_db),
):
    try:
        # Validate input parameters
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        if not isinstance(dc_id, int) or dc_id <= 0:
            raise HTTPException(status_code=400, detail="Document collection ID must be a positive integer")
        if not isinstance(stem, bool):
            raise HTTPException(status_code=400, detail="Stem must be a boolean value")
        if not isinstance(stopword, bool):
            raise HTTPException(status_code=400, detail="Stopword must be a boolean value")
        if query_tf not in ["raw", "log", "augmented", "binary"]:
            raise HTTPException(status_code=400, detail="Invalid query TF type")
        if doc_tf not in ["raw", "log", "augmented", "binary"]:
            raise HTTPException(status_code=400, detail="Invalid document TF type")
        if not isinstance(query_idf, bool):
            raise HTTPException(status_code=400, detail="Query IDF must be a boolean value")
        if not isinstance(query_norm, bool):
            raise HTTPException(status_code=400, detail="Query normalization must be a boolean value")
        if not isinstance(doc_idf, bool):
            raise HTTPException(status_code=400, detail="Document IDF must be a boolean value")
        if not isinstance(doc_norm, bool):
            raise HTTPException(status_code=400, detail="Document normalization must be a boolean value")
        if not isinstance(synset, list):
            raise HTTPException(status_code=400, detail="Synset must be a list of strings")

        for s in synset:
            if s not in VALID_SYNSET_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid synset type: '{s}'. Allowed types: {', '.join(sorted(VALID_SYNSET_TYPES))}"
                )
            
        results = await search_query(
            db=db,
            dc_id=dc_id,
            query=query,
            synset=synset,
            stem=stem,
            stopword=stopword,
            query_tf=query_tf,
            query_idf=query_idf,
            query_norm=query_norm,
            doc_tf=doc_tf,
            doc_idf=doc_idf,
            doc_norm=doc_norm
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search-batch/")
async def search_batch(
    dc_id: int = Form(...),
    queries: UploadFile = File(...),
    relevance: UploadFile = File(...),
    settings: UploadFile = File(...),
    filename: str = Form(...),
    download: bool = Form(False), 
    db: AsyncSession = Depends(get_db)
):
    """Batch search endpoint with MAP evaluation and formatted output"""
    
    try:
        start_parse = time.time()        
        print(f"[DEBUG] Starting batch search with dc_id: {dc_id}, filename: {filename}")

        # Read and parse input files
        queries_content = (await queries.read()).decode('utf-8')
        relevance_content = (await relevance.read()).decode('utf-8')
        settings_content = (await settings.read()).decode('utf-8')

        # print(f"[DEBUG] File contents length - Queries: {len(queries_content)}, Relevance: {len(relevance_content)}, Settings: {len(settings_content)}")

        if not queries_content or not relevance_content or not settings_content:
            raise HTTPException(status_code=400, detail="Input files cannot be empty")
        if not isinstance(dc_id, int) or dc_id <= 0:
            raise HTTPException(status_code=400, detail="Document collection ID must be a positive integer")
        if not filename:
            raise HTTPException(status_code=400, detail="Filename cannot be empty")
        
        try:
            print("[DEBUG] Parsing queries file...")
            parsed_queries = parse_queries_file(queries_content)
            # print(f"[DEBUG] Parsed {len(parsed_queries)} queries")
            # for qid, qtext in list(parsed_queries.items())[:3]: 
            #     print(f"[DEBUG] Query {qid}: {qtext[:100]}...")
        except Exception as e:
            print(f"[ERROR] Failed to parse queries file: {e}")
            raise HTTPException(status_code=400, detail=f"Error parsing queries file: {str(e)}")

        try:
            print("[DEBUG] Parsing relevance file...")
            parsed_relevance = parse_relevance_file(relevance_content)
            # print(f"[DEBUG] Parsed relevance for {len(parsed_relevance)} queries")
            # for qid, docs in list(parsed_relevance.items())[:3]:
            #     print(f"[DEBUG] Query {qid} relevant docs: {list(docs)[:5]}...")
        except Exception as e:
            print(f"[ERROR] Failed to parse relevance file: {e}")
            raise HTTPException(status_code=400, detail=f"Error parsing relevance file: {str(e)}")

        try:
            print("[DEBUG] Parsing settings file...")
            # print(f"[DEBUG] Settings content preview:\n{settings_content[:500]}")
            parsed_settings = parse_settings_file(settings_content)
            # print(f"[DEBUG] Parsed settings for {len(parsed_settings)} configurations")
            # for sid, settings in parsed_settings.items():
            #     print(f"[DEBUG] Settings ID {sid}: {settings}")
        except Exception as e:
            print(f"[ERROR] Failed to parse settings file: {e}")
            print(f"[ERROR] Settings file content:\n{settings_content}")
            raise HTTPException(status_code=400, detail=f"Error parsing settings file: {str(e)}")

        try:
            print("[DEBUG] Creating query input objects...")
            queries = query_input(
                dc_id,
                parsed_queries,
                parsed_relevance,
                parsed_settings
            )
            print(f"[DEBUG] Created {len(queries)} query input objects")
            
            # # Debug first query object
            # if queries:
            #     first_query = queries[0]
            #     print(f"[DEBUG] First query object:")
            #     print(f"  - query_id: {first_query.query_id}")
            #     print(f"  - query_text: {first_query.query_text[:100]}...")
            #     print(f"  - relevant_docs: {list(first_query.relevant_docs)[:5]}...")
            #     print(f"  - settings: {first_query.settings}")
        except Exception as e:
            print(f"[ERROR] Failed to create query input objects: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Error creating query objects: {str(e)}")
        
        elapsed_parsing_time = time.time() - start_parse
        start_search = time.time()
        try:
            print("[DEBUG] Starting batch search...")
            results = await search_query_batch(
                db=db,
                dc_id=dc_id,
                queries=queries,
                parallel=True
            )
            print(f"[DEBUG] Batch search completed successfully")
            print(f"[DEBUG] MAP Initial: {results['map_initial']:.4f}")
            print(f"[DEBUG] MAP Expanded: {results['map_expanded']:.4f}")
            print(f"[DEBUG] Processed queries: {results['processed_queries']}")
        except Exception as e:
            print(f"[ERROR] Failed during batch search: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error during batch search: {str(e)}")
        
        elapsed_search_time = time.time() - start_search
        print("[DEBUG] Parsing Time:", elapsed_parsing_time)
        print("[DEBUG] Searching Time:", elapsed_search_time)

        if download:
            response_data = {
                'map_initial': results['map_initial'],
                'map_expanded': results['map_expanded'],
                'query_results': results['query_results'],
                'processed_queries': results['processed_queries'],
                'filename': f"{filename}.txt"
            }
                    
            file_content = results['download_content']
            file_name = f"{filename}.txt"
            
            return Response(
                content=file_content,
                media_type='text/plain',
                headers={
                    'Content-Disposition': f'attachment; filename="{file_name}"',
                    'Content-Type': 'text/plain; charset=utf-8',
                    'X-Response-Data': json.dumps(response_data)
                }
            )
        
        return {
            'map_initial': results['map_initial'],
            'map_expanded': results['map_expanded'],
            'query_results': results['query_results'],
            'processed_queries': results['processed_queries'],
            'download_content': results['download_content'],
            'filename': f"{filename}.txt"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error in search_batch: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")