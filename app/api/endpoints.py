import os
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import Optional, List
import json

from app.api.models import (
    DocumentUploadResponse,
    DocumentInfo,
    SearchRequest,
    SearchResponse,
    SearchResult,
    ChatRequest,
    ChatResponse,
    SourceInfo,
    StatsResponse,
    DeleteResponse
)
from app.core.rag_pipeline import rag_pipeline
from app.core.config import config

app = FastAPI(
    title="ILIN API",
    description="Integrated Localized Intelligence Node - Offline RAG API",
    version="1.0.0"
)

DOCUMENTS_PATH = Path(config.documents_path)
DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)


@app.get("/")
def root():
    return {
        "message": "ILIN - Integrated Localized Intelligence Node",
        "status": "online",
        "version": "1.0.0"
    }


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in config.supported_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported: {config.supported_extensions}"
            )
        
        file_path = DOCUMENTS_PATH / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        processed_doc = rag_pipeline.ingest_document(str(file_path))
        
        return DocumentUploadResponse(
            success=True,
            message=f"Successfully processed {file.filename}",
            document_id=processed_doc.document_id,
            filename=processed_doc.filename,
            total_chunks=processed_doc.total_chunks
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    try:
        documents = rag_pipeline.list_documents()
        return [DocumentInfo(**doc) for doc in documents]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
def delete_document(document_id: str):
    try:
        success = rag_pipeline.delete_document(document_id)
        if success:
            return DeleteResponse(
                success=True,
                message=f"Document {document_id} deleted successfully",
                document_id=document_id
            )
        else:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    try:
        results = rag_pipeline.search_only(
            query=request.query,
            search_mode=request.search_mode,
            top_k=request.top_k
        )
        
        search_results = [
            SearchResult(
                rank=r['rank'],
                chunk_id=r['chunk_id'],
                document_id=r['document_id'],
                content=r['content'],
                score=r['score'],
                search_type=r['search_type'],
                source=r['source']
            )
            for r in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        response = rag_pipeline.query(
            question=request.question,
            search_mode=request.search_mode,
            top_k=request.top_k,
            stream=False
        )
        
        sources = [SourceInfo(**s) for s in response['sources']]
        
        return ChatResponse(
            answer=response['answer'],
            sources=sources,
            model=response.get('model', config.ollama_model)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    try:
        response_data = rag_pipeline.query(
            question=request.question,
            search_mode=request.search_mode,
            top_k=request.top_k,
            stream=True
        )
        
        def generate():
            for chunk in response_data['stream']:
                data = {
                    'content': chunk.content,
                    'done': chunk.done
                }
                if chunk.done:
                    data['sources'] = response_data['sources']
                yield f"data: {json.dumps(data)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    try:
        stats = rag_pipeline.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ILIN"}
