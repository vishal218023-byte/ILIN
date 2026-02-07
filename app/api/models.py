from typing import List, Optional
from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    document_id: Optional[str] = None
    filename: Optional[str] = None
    total_chunks: Optional[int] = None


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size: int
    total_chunks: int
    created_at: str


class SearchResult(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    content: str
    score: float
    search_type: str
    source: str


class SearchRequest(BaseModel):
    query: str
    search_mode: Optional[str] = "hybrid"
    top_k: Optional[int] = 10


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int


class ChatRequest(BaseModel):
    question: str
    search_mode: Optional[str] = "hybrid"
    top_k: Optional[int] = 10


class SourceInfo(BaseModel):
    document_id: str
    filename: str
    relevance_score: float
    search_type: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    model: str


class StatsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    index_size: int
    embedding_dimension: int
    index_path: str
    metadata_db: str


class DeleteResponse(BaseModel):
    success: bool
    message: str
    document_id: str
