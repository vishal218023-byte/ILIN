import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import math

from app.core.config import config
from app.core.embedding_engine import embedding_engine
from app.core.vector_store import vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int
    search_type: str


class BM25Index:
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.term_freqs: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.total_docs: int = 0
        self.k1: float = 1.5
        self.b: float = 0.75
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        for doc in documents:
            idx = len(self.documents)
            self.documents.append(doc)
            
            tokens = self._tokenize(doc.get('content', ''))
            self.doc_lengths.append(len(tokens))
            
            term_counts = defaultdict(int)
            for token in tokens:
                term_counts[token] += 1
            
            for term, count in term_counts.items():
                self.term_freqs[term][idx] = count
                self.doc_freqs[term] += 1
        
        self.total_docs = len(self.documents)
        self.avg_doc_length = sum(self.doc_lengths) / max(self.total_docs, 1)
    
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if self.total_docs == 0:
            return []
        
        query_tokens = self._tokenize(query)
        scores = defaultdict(float)
        
        for token in query_tokens:
            if token not in self.term_freqs:
                continue
            
            idf = math.log(
                (self.total_docs - self.doc_freqs[token] + 0.5) / 
                (self.doc_freqs[token] + 0.5) + 1
            )
            
            for doc_idx, tf in self.term_freqs[token].items():
                doc_len = self.doc_lengths[doc_idx]
                tf_component = tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                )
                scores[doc_idx] += idf * tf_component
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]


class Retriever:
    def __init__(
        self,
        search_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        enable_reranking: Optional[bool] = None
    ):
        self.search_mode = search_mode or config.search_mode
        self.top_k = top_k or config.top_k
        self.enable_reranking = enable_reranking if enable_reranking is not None else config.get('retrieval.enable_reranking', True)
        self.hybrid_alpha = config.get('retrieval.hybrid_alpha', 0.7)
        self.min_score_threshold = config.get('retrieval.min_score_threshold', 0.5)
        
        self.bm25_index = BM25Index()
        self._build_bm25_index()
        
        self.reranker = None
        if self.enable_reranking:
            self._load_reranker()
    
    def _build_bm25_index(self):
        try:
            all_metadata = vector_store.metadata
            if all_metadata:
                self.bm25_index.add_documents(all_metadata)
                logger.info(f"Built BM25 index with {len(all_metadata)} documents")
        except Exception as e:
            logger.error(f"Error building BM25 index: {str(e)}")
    
    def _load_reranker(self):
        try:
            from sentence_transformers import CrossEncoder
            rerank_model = config.get('retrieval.rerank_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.reranker = CrossEncoder(rerank_model)
            logger.info(f"Loaded reranker model: {rerank_model}")
        except ImportError:
            logger.warning("Could not load reranker. Install sentence-transformers.")
            self.reranker = None
        except Exception as e:
            logger.error(f"Error loading reranker: {str(e)}")
            self.reranker = None
    
    def search(
        self,
        query: str,
        search_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        mode = search_mode or self.search_mode
        k = top_k or self.top_k
        
        if mode == 'semantic':
            results = self._semantic_search(query, k, filter_dict)
        elif mode == 'keyword':
            results = self._keyword_search(query, k, filter_dict)
        elif mode == 'hybrid':
            results = self._hybrid_search(query, k, filter_dict)
        else:
            raise ValueError(f"Unknown search mode: {mode}")
        
        if self.enable_reranking and self.reranker and len(results) > 0:
            results = self._rerank_results(query, results, min(k, len(results)))
        
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _semantic_search(
        self,
        query: str,
        top_k: int,
        filter_dict: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        query_embedding = embedding_engine.embed_query(query)
        
        raw_results = vector_store.search(
            query_embedding,
            top_k=top_k * 2,
            filter_dict=filter_dict
        )
        
        results = []
        for raw in raw_results:
            if raw['score'] >= self.min_score_threshold:
                results.append(SearchResult(
                    chunk_id=raw['chunk_id'],
                    document_id=raw['document_id'],
                    content=raw['content'],
                    score=raw['score'],
                    metadata=raw['metadata'],
                    rank=0,
                    search_type='semantic'
                ))
        
        return results[:top_k]
    
    def _keyword_search(
        self,
        query: str,
        top_k: int,
        filter_dict: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        if len(self.bm25_index.documents) == 0:
            return []
        
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
        
        results = []
        for doc_idx, score in bm25_results:
            if score <= 0:
                continue
                
            doc = self.bm25_index.documents[doc_idx]
            
            if filter_dict:
                match = all(doc.get(k) == v for k, v in filter_dict.items())
                if not match:
                    continue
            
            normalized_score = min(score / 10.0, 1.0)
            
            results.append(SearchResult(
                chunk_id=doc.get('chunk_id', f"chunk_{doc_idx}"),
                document_id=doc.get('document_id', ''),
                content=doc.get('content', ''),
                score=normalized_score,
                metadata=doc,
                rank=0,
                search_type='keyword'
            ))
        
        return results[:top_k]
    
    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        filter_dict: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        semantic_results = self._semantic_search(query, top_k * 2, filter_dict)
        keyword_results = self._keyword_search(query, top_k * 2, filter_dict)
        
        combined_scores: Dict[str, Dict[str, Any]] = {}
        
        for result in semantic_results:
            cid = result.chunk_id
            if cid not in combined_scores:
                combined_scores[cid] = {
                    'result': result,
                    'semantic_score': result.score,
                    'keyword_score': 0.0
                }
        
        for result in keyword_results:
            cid = result.chunk_id
            if cid in combined_scores:
                combined_scores[cid]['keyword_score'] = result.score
            else:
                combined_scores[cid] = {
                    'result': result,
                    'semantic_score': 0.0,
                    'keyword_score': result.score
                }
        
        final_results = []
        for cid, data in combined_scores.items():
            hybrid_score = (
                self.hybrid_alpha * data['semantic_score'] +
                (1 - self.hybrid_alpha) * data['keyword_score']
            )
            
            result = data['result']
            result.score = hybrid_score
            result.search_type = 'hybrid'
            final_results.append(result)
        
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]
    
    def _rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        if not self.reranker or len(results) == 0:
            return results
        
        try:
            pairs = [[query, result.content] for result in results]
            rerank_scores = self.reranker.predict(pairs)
            
            for result, score in zip(results, rerank_scores):
                result.score = float(score)
            
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            return results[:top_k]
    
    def refresh_index(self):
        self.bm25_index = BM25Index()
        self._build_bm25_index()
        logger.info("Retriever index refreshed")


retriever = Retriever()
