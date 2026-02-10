import os
import pickle
import logging
from typing import List, Optional, Dict, Any
import numpy as np
from pathlib import Path

from app.core.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingEngine:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        cache_dir: str = "models"
    ):
        self.model_name = model_name or config.embedding_model
        self.device = device or config.embedding_device
        self.batch_size = batch_size or config.embedding_batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.embedding_dim = None
        self._load_model()
    
    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            cache_path = self.cache_dir / self.model_name.replace('/', '_')
            
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(cache_path),
                device=self.device
            )
            
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        if not texts:
            return np.array([])
        
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        return self.embed_texts([query], normalize=normalize)[0]
    
    def embed_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "content",
        show_progress: bool = True
    ) -> tuple:
        texts = [doc[text_key] for doc in documents]
        embeddings = self.embed_texts(texts, show_progress=show_progress)
        return embeddings, documents
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dim
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = np.dot(document_embeddings, query_embedding.T).flatten()
        return similarities
    
    def save_cache(self, cache_path: str, embeddings: np.ndarray, metadata: List[Dict]):
        cache_data = {
            'embeddings': embeddings,
            'metadata': metadata,
            'model_name': self.model_name
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Embedding cache saved to {cache_path}")
    
    def load_cache(self, cache_path: str) -> tuple:
        if not os.path.exists(cache_path):
            return None, None
        
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        if cache_data.get('model_name') != self.model_name:
            logger.warning("Cache was generated with a different model. Ignoring cache.")
            return None, None
        
        logger.info(f"Embedding cache loaded from {cache_path}")
        return cache_data['embeddings'], cache_data['metadata']


embedding_engine = EmbeddingEngine()
