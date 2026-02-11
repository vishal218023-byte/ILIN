import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to app/config.yaml relative to this file
            config_path = str(Path(__file__).parent.parent / "config.yaml")
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
    
    def set(self, key: str, value: Any):
        keys = key.split('.')
        target = self.config
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value
    
    def save(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
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

    @property
    def groq_api_key(self) -> str:
        """Get Groq API key from environment variable."""
        return os.getenv('GROQ_API_KEY', '')

    @property
    def nvidia_api_key(self) -> str:
        """Get Nvidia API key from environment variable."""
        return os.getenv('NVIDIA_API_KEY', '')


config = Config()
