import os
import yaml
from pathlib import Path
from typing import Dict, Any, List


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    @property
    def embedding_model(self) -> str:
        return self.get('embedding.model', 'all-MiniLM-L6-v2')
    
    @property
    def embedding_batch_size(self) -> int:
        return self.get('embedding.batch_size', 32)
    
    @property
    def embedding_device(self) -> str:
        return self.get('embedding.device', 'cpu')
    
    @property
    def vector_index_path(self) -> str:
        return self.get('vector_store.index_path', 'data/vector_indices/')
    
    @property
    def metadata_db_path(self) -> str:
        return self.get('vector_store.metadata_db', 'data/metadata.db')
    
    @property
    def ollama_base_url(self) -> str:
        return self.get('ollama.base_url', 'http://localhost:11434')
    
    @property
    def ollama_model(self) -> str:
        return self.get('ollama.model', 'llama3.2:3b')
    
    @property
    def chunk_size(self) -> int:
        return self.get('chunking.size', 512)
    
    @property
    def chunk_overlap(self) -> int:
        return self.get('chunking.overlap', 50)
    
    @property
    def search_mode(self) -> str:
        return self.get('retrieval.search_mode', 'hybrid')
    
    @property
    def top_k(self) -> int:
        return self.get('vector_store.top_k', 10)
    
    @property
    def documents_path(self) -> str:
        return self.get('documents.upload_path', 'data/documents/')
    
    @property
    def supported_extensions(self) -> List[str]:
        return self.get('documents.supported_extensions', ['.txt', '.pdf', '.docx', '.md'])


config = Config()
