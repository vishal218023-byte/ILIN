import os
import json
import pickle
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from app.core.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        index_path: Optional[str] = None,
        metadata_db_path: Optional[str] = None,
        embedding_dim: int = 384
    ):
        self.index_path = Path(index_path or config.vector_index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_db_path = metadata_db_path or config.metadata_db_path
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.chunk_id_to_idx: Dict[str, int] = {}
        self.document_chunks: Dict[str, List[str]] = {}
        
        self._init_metadata_db()
        self._load_or_create_index()
    
    def _init_metadata_db(self):
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_type TEXT,
                file_size INTEGER,
                total_chunks INTEGER,
                content_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                content TEXT NOT NULL,
                chunk_index INTEGER,
                start_char INTEGER,
                end_char INTEGER,
                char_count INTEGER,
                created_at TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(document_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Metadata database initialized")
    
    def _load_or_create_index(self):
        index_file = self.index_path / "faiss_index.bin"
        metadata_file = self.index_path / "metadata.pkl"
        
        try:
            import faiss
            
            if index_file.exists() and metadata_file.exists():
                logger.info("Loading existing FAISS index...")
                self.index = faiss.read_index(str(index_file))
                
                with open(metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                
                self._rebuild_mappings()
                logger.info(f"Loaded index with {len(self.metadata)} chunks")
            else:
                logger.info("Creating new FAISS index...")
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                
        except ImportError:
            logger.error("FAISS not installed. Install with: pip install faiss-cpu")
            raise
    
    def _rebuild_mappings(self):
        self.chunk_id_to_idx.clear()
        self.document_chunks.clear()
        
        for idx, meta in enumerate(self.metadata):
            chunk_id = meta.get('chunk_id')
            doc_id = meta.get('document_id')
            
            if chunk_id:
                self.chunk_id_to_idx[chunk_id] = idx
            
            if doc_id:
                if doc_id not in self.document_chunks:
                    self.document_chunks[doc_id] = []
                self.document_chunks[doc_id].append(chunk_id)
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        document_info: Optional[Dict[str, Any]] = None
    ):
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")
        
        if len(chunks) == 0:
            return
        
        try:
            import faiss
            
            start_idx = len(self.metadata)
            
            embeddings = embeddings.astype('float32')
            self.index.add(embeddings)
            
            for i, chunk in enumerate(chunks):
                idx = start_idx + i
                self.metadata.append(chunk)
                
                chunk_id = chunk.get('chunk_id')
                doc_id = chunk.get('document_id')
                
                if chunk_id:
                    self.chunk_id_to_idx[chunk_id] = idx
                
                if doc_id:
                    if doc_id not in self.document_chunks:
                        self.document_chunks[doc_id] = []
                    self.document_chunks[doc_id].append(chunk_id)
            
            if document_info:
                self._save_document_metadata(document_info, chunks)
            
            self._persist_index()
            
            logger.info(f"Added {len(chunks)} chunks to index. Total: {len(self.metadata)}")
            
        except Exception as e:
            logger.error(f"Error adding chunks: {str(e)}")
            raise
    
    def _save_document_metadata(self, document_info: Dict[str, Any], chunks: List[Dict]):
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO documents 
            (document_id, filename, file_path, file_type, file_size, total_chunks, content_hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            document_info.get('document_id'),
            document_info.get('filename'),
            document_info.get('file_path'),
            document_info.get('file_type'),
            document_info.get('file_size'),
            document_info.get('total_chunks'),
            document_info.get('content_hash'),
            document_info.get('created_at', datetime.now().isoformat())
        ))
        
        for chunk in chunks:
            cursor.execute('''
                INSERT OR REPLACE INTO chunks
                (chunk_id, document_id, content, chunk_index, start_char, end_char, char_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                chunk.get('chunk_id'),
                chunk.get('document_id'),
                chunk.get('content'),
                chunk.get('chunk_index'),
                chunk.get('start_char'),
                chunk.get('end_char'),
                chunk.get('char_count'),
                chunk.get('created_at')
            ))
        
        conn.commit()
        conn.close()
    
    def remove_document(self, document_id: str):
        if document_id not in self.document_chunks:
            logger.warning(f"Document {document_id} not found in index")
            return
        
        try:
            chunk_ids = self.document_chunks[document_id]
            indices_to_remove = sorted(
                [self.chunk_id_to_idx[cid] for cid in chunk_ids if cid in self.chunk_id_to_idx],
                reverse=True
            )
            
            for idx in indices_to_remove:
                del self.metadata[idx]
            
            self._rebuild_index_from_metadata()
            self._rebuild_mappings()
            self._persist_index()
            
            conn = sqlite3.connect(self.metadata_db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            conn.commit()
            conn.close()
            
            logger.info(f"Removed document {document_id} with {len(chunk_ids)} chunks")
            
        except Exception as e:
            logger.error(f"Error removing document: {str(e)}")
            raise
    
    def _rebuild_index_from_metadata(self):
        try:
            import faiss
            
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            if self.metadata:
                all_embeddings = []
                for meta in self.metadata:
                    if 'embedding' in meta:
                        all_embeddings.append(meta['embedding'])
                
                if all_embeddings:
                    embeddings_array = np.array(all_embeddings).astype('float32')
                    self.index.add(embeddings_array)
                    
        except Exception as e:
            logger.error(f"Error rebuilding index: {str(e)}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        
        try:
            query_embedding = query_embedding.astype('float32').reshape(1, -1)
            
            scores, indices = self.index.search(query_embedding, min(top_k * 2, self.index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                
                meta = self.metadata[idx]
                
                if filter_dict:
                    match = all(meta.get(k) == v for k, v in filter_dict.items())
                    if not match:
                        continue
                
                result = {
                    'chunk_id': meta.get('chunk_id'),
                    'document_id': meta.get('document_id'),
                    'content': meta.get('content'),
                    'metadata': meta,
                    'score': float(score),
                    'index': int(idx)
                }
                results.append(result)
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            raise
    
    def _persist_index(self):
        index_file = self.index_path / "faiss_index.bin"
        metadata_file = self.index_path / "metadata.pkl"
        
        try:
            import faiss
            faiss.write_index(self.index, str(index_file))
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
                
        except Exception as e:
            logger.error(f"Error persisting index: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_documents': doc_count,
            'total_chunks': chunk_count,
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dimension': self.embedding_dim,
            'index_path': str(self.index_path),
            'metadata_db': self.metadata_db_path
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT document_id, filename, file_type, file_size, total_chunks, created_at, file_path
            FROM documents
            ORDER BY created_at DESC
        ''')
        
        columns = [description[0] for description in cursor.description]
        documents = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return documents

    def get_document_path(self, document_id: str) -> Optional[str]:
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT file_path FROM documents WHERE document_id = ?", (document_id,))
        result = cursor.fetchone()
        
        conn.close()
        return result[0] if result else None


vector_store = VectorStore()
