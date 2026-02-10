import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.core.config import config
from app.core.indexing.document_processor import DocumentProcessor, ProcessedDocument
from app.core.retrieval.embedding_engine import embedding_engine
from app.core.retrieval.vector_store import vector_store
from app.core.retrieval.retriever import retriever, SearchResult
from app.core.llm.llm_client import get_llm_client, ChatResponse, RAGContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self.document_processor = DocumentProcessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    def ingest_document(self, file_path: str, document_id: Optional[str] = None) -> ProcessedDocument:
        logger.info(f"Ingesting document: {file_path}")
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path_obj.suffix.lower()
        if file_ext not in config.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        processed_doc = self.document_processor.process_file(file_path, document_id)
        
        if not processed_doc.chunks:
            logger.warning(f"No chunks extracted from {file_path}")
            return processed_doc
        
        chunk_dicts = []
        for chunk in processed_doc.chunks:
            chunk_dict = {
                'chunk_id': chunk.chunk_id,
                'document_id': chunk.document_id,
                'content': chunk.content,
                'chunk_index': chunk.metadata.get('chunk_index'),
                'start_char': chunk.metadata.get('start_char'),
                'end_char': chunk.metadata.get('end_char'),
                'char_count': chunk.metadata.get('char_count'),
                'source': chunk.metadata.get('source'),
                'created_at': chunk.metadata.get('created_at')
            }
            chunk_dicts.append(chunk_dict)
        
        texts = [chunk.content for chunk in processed_doc.chunks]
        embeddings = embedding_engine.embed_texts(texts, show_progress=True)
        
        document_info = {
            'document_id': processed_doc.document_id,
            'filename': processed_doc.filename,
            'file_path': processed_doc.file_path,
            'file_type': processed_doc.file_type,
            'file_size': processed_doc.file_size,
            'total_chunks': processed_doc.total_chunks,
            'content_hash': processed_doc.content_hash,
            'created_at': processed_doc.created_at.isoformat()
        }
        
        vector_store.add_chunks(chunk_dicts, embeddings, document_info)
        
        retriever.refresh_index()
        
        logger.info(f"Successfully ingested document: {processed_doc.filename} with {processed_doc.total_chunks} chunks")
        return processed_doc
    
    def query(
        self,
        question: str,
        search_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        logger.info(f"Processing query (RAG={use_rag}): {question}")
        
        if not use_rag:
            # Direct chat with LLM
            if stream:
                return {
                    'stream': get_llm_client().generate(question, stream=True),
                    'sources': [],
                    'search_results': []
                }
            else:
                response = get_llm_client().generate(question, stream=False)
                return {
                    'answer': response.content,
                    'sources': [],
                    'search_results': [],
                    'model': response.model
                }

        # RAG Logic
        search_results = retriever.search(
            query=question,
            search_mode=search_mode,
            top_k=top_k or config.top_k
        )
        
        if not search_results:
            return {
                'answer': "I don't have enough information to answer this question based on the available documents.",
                'sources': [],
                'search_results': []
            }
        
        contexts = [
            RAGContext(
                content=result.content,
                source=result.metadata.get('source', 'Unknown'),
                score=result.score,
                document_id=result.document_id
            )
            for result in search_results
        ]
        
        if stream:
            return {
                'stream': get_llm_client().chat_with_rag(question, contexts, stream=True),
                'sources': self._format_sources(search_results),
                'search_results': self._format_search_results(search_results)
            }
        else:
            response = get_llm_client().chat_with_rag(question, contexts, stream=False)
            
            return {
                'answer': response.content,
                'sources': self._format_sources(search_results),
                'search_results': self._format_search_results(search_results),
                'model': response.model
            }

    
    def search_only(
        self,
        query: str,
        search_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        search_results = retriever.search(
            query=query,
            search_mode=search_mode,
            top_k=top_k or config.top_k
        )
        
        if filters:
            search_results = [
                r for r in search_results
                if all(r.metadata.get(k) == v for k, v in filters.items())
            ]
        
        return self._format_search_results(search_results)
    
    def delete_document(self, document_id: str) -> bool:
        try:
            vector_store.remove_document(document_id)
            retriever.refresh_index()
            logger.info(f"Deleted document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            return False
    
    def list_documents(self) -> List[Dict[str, Any]]:
        return vector_store.list_documents()
    
    def get_stats(self) -> Dict[str, Any]:
        return vector_store.get_stats()
    
    def _format_sources(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        sources = []
        seen_docs = set()
        
        for result in results:
            doc_id = result.document_id
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                sources.append({
                    'document_id': doc_id,
                    'filename': result.metadata.get('source', 'Unknown').split('/')[-1],
                    'relevance_score': result.score,
                    'search_type': result.search_type
                })
        
        return sources
    
    def _format_search_results(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        return [
            {
                'rank': result.rank,
                'chunk_id': result.chunk_id,
                'document_id': result.document_id,
                'content': result.content[:500] + "..." if len(result.content) > 500 else result.content,
                'score': round(result.score, 4),
                'search_type': result.search_type,
                'source': result.metadata.get('source', 'Unknown')
            }
            for result in results
        ]


rag_pipeline = RAGPipeline()
