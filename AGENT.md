# AGENT.md - Coding Agent Guide for ILIN Project

## Project Overview

**ILIN (Integrated Localized Intelligence Node)** is an offline, air-gapped RAG (Retrieval-Augmented Generation) system designed for secure local document processing and AI-powered question answering. The system operates completely offline after initial setup, making it ideal for secure environments.

### Core Purpose
- Local document ingestion and indexing
- Semantic search with multiple retrieval modes
- AI-powered chat interface using local LLM
- RESTful API for programmatic access
- Web-based UI for user interaction

## Technology Stack

### Backend Framework
- **FastAPI** (0.104.1) - REST API server
- **Uvicorn** (0.24.0) - ASGI server

### Frontend
- **Streamlit** (1.28.1) - Web-based UI framework
- Custom HTML/CSS styling embedded in Streamlit components

### AI/ML Components
- **sentence-transformers** (2.2.2) - Text embeddings (model: `all-MiniLM-L6-v2`)
- **FAISS** (faiss-cpu 1.7.4) - Vector similarity search and indexing
- **Ollama** - Local LLM inference (default model: `llama3.2:3b`)
- **Cross-encoder** - Reranking search results (ms-marco-MiniLM-L-6-v2)

### Document Processing
- **PyPDF2** (3.0.1) - PDF text extraction
- **python-docx** (1.1.0) - DOCX processing
- **beautifulsoup4** (4.12.2) - HTML parsing
- Built-in CSV support

### Search & Retrieval
- **rank-bm25** (0.2.2) - Keyword search algorithm
- Custom hybrid search combining semantic + keyword

### Data Management
- **SQLite** - Metadata storage (via Python stdlib)
- **FAISS** - Vector indices
- **Pickle** - Index serialization

### Supporting Libraries
- **Pydantic** (2.5.0) - Data validation and models
- **PyYAML** (6.0.1) - Configuration management
- **NumPy** (1.26.2) - Numerical operations
- **scikit-learn** (1.3.2) - ML utilities
- **requests** (2.31.0) - HTTP client for Ollama API

## Project Architecture

### Directory Structure
```
ILIN/
├── app/
│   ├── __init__.py
│   ├── api/                    # FastAPI REST API
│   │   ├── __init__.py
│   │   ├── endpoints.py        # API route handlers
│   │   └── models.py           # Pydantic request/response models
│   ├── core/                   # Core business logic
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration loader
│   │   ├── document_processor.py  # Document parsing & chunking
│   │   ├── embedding_engine.py    # Embedding generation
│   │   ├── ollama_client.py       # LLM client
│   │   ├── rag_pipeline.py        # RAG orchestration
│   │   ├── retriever.py           # Search engine
│   │   └── vector_store.py        # FAISS wrapper
│   └── ui/                     # Streamlit UI
│       ├── __init__.py
│       └── streamlit_app.py
├── data/                       # Runtime data (created automatically)
│   ├── documents/              # Uploaded documents
│   ├── vector_indices/         # FAISS index files
│   └── metadata.db            # SQLite database
├── models/                     # Downloaded ML models (created on first run)
├── config.yaml                # Main configuration file
├── requirements.txt           # Python dependencies
├── run_api.py                 # API server launcher
├── run_ui.py                  # UI launcher
├── run_api.bat               # Windows API launcher
├── run_ui.bat                # Windows UI launcher
├── run_both.bat              # Windows dual launcher
├── setup.bat                 # Windows setup script
├── START.bat                 # Windows quick start
└── README.md                 # User documentation
```

### Component Responsibilities

#### 1. **Core Layer** (`app/core/`)

**config.py**
- Loads and parses `config.yaml`
- Provides property-based access to configuration values
- Singleton pattern via `config = Config()` at module level

**document_processor.py**
- Extracts text from multiple file formats (PDF, DOCX, TXT, MD, HTML, CSV)
- Implements intelligent text chunking with overlap
- Creates `DocumentChunk` and `ProcessedDocument` dataclasses
- Generates unique document IDs and content hashes
- Uses multiple encoding fallbacks for text files

**embedding_engine.py**
- Manages sentence-transformer model lifecycle
- Generates embeddings for text and queries
- Batch processing with configurable batch size
- Caching support for embeddings
- Singleton pattern via `embedding_engine = EmbeddingEngine()` at module level

**vector_store.py**
- FAISS index management (IndexFlatIP for inner product)
- SQLite integration for metadata storage
- Document and chunk CRUD operations
- Automatic index persistence and recovery
- Maintains in-memory mappings for fast lookups

**retriever.py**
- Three search modes: semantic, keyword (BM25), hybrid
- BM25 index for keyword search
- Cross-encoder reranking support
- Score normalization and combination
- Minimum score thresholding
- Singleton pattern via `retriever = Retriever()` at module level

**ollama_client.py**
- HTTP client for local Ollama server
- RAG-specific prompt templating
- Streaming and non-streaming response handling
- Context window management
- Token approximation for context fitting
- Singleton pattern via `ollama_client = OllamaClient()` at module level

**rag_pipeline.py**
- High-level RAG orchestration
- Document ingestion workflow
- Query processing pipeline
- Source formatting and deduplication
- Stats and document management
- Singleton pattern via `rag_pipeline = RAGPipeline()` at module level

#### 2. **API Layer** (`app/api/`)

**models.py**
- Pydantic models for request/response validation
- All models inherit from `BaseModel`
- Optional fields use `Optional[Type]` with defaults

**endpoints.py**
- FastAPI application instance
- RESTful endpoints for all operations
- Error handling with HTTPException
- File upload handling
- Streaming response support for chat

#### 3. **UI Layer** (`app/ui/`)

**streamlit_app.py**
- Multi-tab interface: Chat, Documents, Search
- Session state management
- Custom CSS styling
- Real-time status indicators
- Direct integration with `rag_pipeline`

## Coding Style & Patterns

### Python Style Guide

1. **Imports**
   - Standard library imports first
   - Third-party imports second
   - Local imports last
   - Alphabetically sorted within groups
   - Example:
     ```python
     import os
     import logging
     from pathlib import Path
     from typing import List, Dict, Any, Optional
     
     import numpy as np
     from sentence_transformers import SentenceTransformer
     
     from app.core.config import config
     ```

2. **Type Hints**
   - All function parameters and return types are typed
   - Use `typing` module: `List`, `Dict`, `Optional`, `Any`, `Tuple`
   - Example:
     ```python
     def process_file(self, file_path: str, document_id: Optional[str] = None) -> ProcessedDocument:
         pass
     ```

3. **Dataclasses**
   - Use `@dataclass` decorator for data containers
   - Prefer dataclasses over dictionaries for structured data
   - Example:
     ```python
     from dataclasses import dataclass
     
     @dataclass
     class DocumentChunk:
         content: str
         metadata: Dict[str, Any]
         chunk_id: str
         document_id: str
     ```

4. **Class Structure**
   - `__init__` with optional parameters using config defaults
   - Private methods prefixed with `_`
   - Public methods for external API
   - Example:
     ```python
     class VectorStore:
         def __init__(self, index_path: Optional[str] = None):
             self.index_path = Path(index_path or config.vector_index_path)
             self._init_metadata_db()
         
         def _init_metadata_db(self):
             # Private helper
             pass
         
         def add_chunks(self, chunks: List[Dict]):
             # Public API
             pass
     ```

5. **Logging**
   - Use Python's `logging` module
   - Logger per module: `logger = logging.getLogger(__name__)`
   - Log levels: INFO for operations, ERROR for failures, WARNING for issues
   - Example:
     ```python
     import logging
     
     logging.basicConfig(level=logging.INFO)
     logger = logging.getLogger(__name__)
     
     logger.info(f"Ingesting document: {file_path}")
     logger.error(f"Error processing: {str(e)}")
     ```

6. **Error Handling**
   - Try-except blocks for external operations
   - Specific exceptions over generic `Exception` where possible
   - Re-raise with context when appropriate
   - Log errors before raising
   - Example:
     ```python
     try:
         result = process_document(path)
     except FileNotFoundError:
         logger.error(f"File not found: {path}")
         raise
     except Exception as e:
         logger.error(f"Unexpected error: {str(e)}")
         raise
     ```

7. **Configuration Access**
   - Use `config` singleton from `app.core.config`
   - Access via property methods: `config.ollama_model`
   - Fallback to `config.get('key.nested', default_value)`

8. **Singleton Pattern**
   - Module-level instantiation for shared components
   - Example:
     ```python
     # At end of module
     vector_store = VectorStore()
     ```

9. **String Formatting**
   - Use f-strings for interpolation: `f"Processing {filename}"`
   - Multi-line strings with triple quotes for prompts/SQL

10. **File Paths**
    - Use `pathlib.Path` for all path operations
    - Create directories with `mkdir(parents=True, exist_ok=True)`
    - Example:
      ```python
      from pathlib import Path
      
      file_path = Path(config.documents_path) / filename
      file_path.parent.mkdir(parents=True, exist_ok=True)
      ```

### Database Patterns

**SQLite**
- Connection per operation (not pooled)
- Context managers for cleanup
- Parameterized queries (never string interpolation)
- Example:
  ```python
  conn = sqlite3.connect(self.metadata_db_path)
  cursor = conn.cursor()
  cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
  conn.commit()
  conn.close()
  ```

**FAISS**
- Load index at initialization
- Persist after modifications
- Use `IndexFlatIP` for inner product similarity
- Convert embeddings to float32
- Example:
  ```python
  import faiss
  
  self.index = faiss.IndexFlatIP(embedding_dim)
  embeddings = embeddings.astype('float32')
  self.index.add(embeddings)
  faiss.write_index(self.index, str(index_file))
  ```

### API Design Patterns

**FastAPI Endpoints**
- Use Pydantic models for request/response
- HTTP status codes via `HTTPException`
- Async handlers for I/O operations (`async def`)
- Path parameters for IDs, body for complex data
- Example:
  ```python
  @app.post("/documents/upload", response_model=DocumentUploadResponse)
  async def upload_document(file: UploadFile = File(...)):
      try:
          result = process_file(file)
          return DocumentUploadResponse(success=True, **result)
      except Exception as e:
          raise HTTPException(status_code=500, detail=str(e))
  ```

**Streaming Responses**
- Generator functions for streaming
- Server-Sent Events (SSE) format: `data: {json}\n\n`
- Example:
  ```python
  def generate():
      for chunk in stream:
          yield f"data: {json.dumps(chunk)}\n\n"
  
  return StreamingResponse(generate(), media_type="text/event-stream")
  ```

### Streamlit Patterns

**Session State**
- Initialize in `init_session_state()`
- Check existence before use
- Example:
  ```python
  if 'chat_history' not in st.session_state:
      st.session_state.chat_history = []
  ```

**Layout**
- Tabs for major sections
- Columns for horizontal layout
- Expanders for collapsible content
- Example:
  ```python
  tab1, tab2 = st.tabs(["Upload", "List"])
  col1, col2 = st.columns([3, 1])
  with st.expander("Details"):
      st.write(info)
  ```

**Custom Styling**
- Inline CSS via `st.markdown(..., unsafe_allow_html=True)`
- CSS classes for reusable styles
- Defined at module level

## Configuration System

### config.yaml Structure
```yaml
embedding:
  model: "all-MiniLM-L6-v2"
  batch_size: 32
  device: "cpu"
  cache_dir: "models"

vector_store:
  index_path: "data/vector_indices/"
  metadata_db: "data/metadata.db"
  top_k: 10
  rerank_top_k: 5

ollama:
  base_url: "http://localhost:11434"
  model: "llama3.2:3b"
  temperature: 0.7
  max_tokens: 2048
  context_window: 4096
  timeout: 120

retrieval:
  search_mode: "hybrid"         # semantic | keyword | hybrid
  hybrid_alpha: 0.7             # Weight for semantic in hybrid
  min_score_threshold: 0.5
  enable_reranking: true
  rerank_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

chunking:
  size: 512                     # Characters per chunk
  overlap: 50                   # Character overlap
  separators:
    - "\n\n"
    - "\n"
    - ". "
    - " "

documents:
  upload_path: "data/documents/"
  supported_extensions:
    - ".txt"
    - ".pdf"
    - ".docx"
    - ".md"
    - ".html"
    - ".csv"
  max_file_size_mb: 50
  auto_reindex: true

ui:
  page_title: "ILIN - Intelligence Node"
  page_icon: "🧠"
  max_upload_size: 50
```

### Accessing Configuration
```python
from app.core.config import config

# Via properties
model = config.ollama_model
chunk_size = config.chunk_size

# Via get method with nested keys
timeout = config.get('ollama.timeout', 120)
separators = config.get('chunking.separators', ['\n'])
```

## Key Algorithms & Workflows

### Document Ingestion Pipeline

1. **Upload** → File saved to `data/documents/`
2. **Text Extraction** → Format-specific parsers (PDF, DOCX, etc.)
3. **Preprocessing** → Whitespace normalization, separator preservation
4. **Chunking** → Sliding window with overlap, boundary detection
5. **Embedding Generation** → Batch processing via sentence-transformers
6. **Vector Storage** → FAISS index insertion
7. **Metadata Storage** → SQLite records
8. **Index Refresh** → BM25 index rebuild

### Chunking Algorithm

```python
# Key characteristics:
- Fixed size with overlap
- Smart boundary detection (prefer sentence/paragraph breaks)
- Fallback separators: "\n\n" → "\n" → ". " → " "
- Metadata tracking: start_char, end_char, chunk_index
- Minimum chunk size enforcement
```

### Search Modes

**Semantic Search**
```
Query → Embedding → FAISS similarity → Top-K results
```

**Keyword Search (BM25)**
```
Query → Tokenization → BM25 scoring → Top-K results
```

**Hybrid Search**
```
Semantic results + Keyword results
→ Score normalization
→ Weighted combination (alpha * semantic + (1-alpha) * keyword)
→ Deduplication
→ Re-ranking (optional)
→ Top-K results
```

**Re-ranking**
```
Top results → Cross-encoder scoring → Re-sorted by new scores
```

### RAG Query Pipeline

1. **Query Reception** → User question received
2. **Retrieval** → Search mode selection → Top-K chunks
3. **Context Formatting** → Chunks with source citations
4. **Context Windowing** → Fit within LLM token limit
5. **Prompt Construction** → Template + context + question
6. **LLM Generation** → Ollama API call (streaming or batch)
7. **Response Formatting** → Answer + source metadata
8. **Return** → Structured response with citations

### Prompt Template

```python
"""You are an intelligent assistant for the ILIN system.
Your task is to provide accurate, helpful answers based on the provided context.

CONTEXT INFORMATION:
{context}

USER QUESTION:
{question}

Instructions:
- Answer using ONLY the information in the context
- If insufficient information, say so explicitly
- Be concise and direct
- Cite source documents
- Provide step-by-step instructions when asked

Your response:"""
```

## Database Schema

### SQLite Tables

**documents**
```sql
CREATE TABLE documents (
    document_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT,
    file_size INTEGER,
    total_chunks INTEGER,
    content_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**chunks**
```sql
CREATE TABLE chunks (
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
```

### FAISS Index Structure

- **Type**: `IndexFlatIP` (Inner Product / Cosine Similarity)
- **Dimension**: 384 (from all-MiniLM-L6-v2)
- **Storage**: `data/vector_indices/faiss_index.bin`
- **Metadata**: `data/vector_indices/metadata.pkl` (pickled list)

### Metadata Structure

```python
# Each entry in metadata list:
{
    'chunk_id': str,           # Unique chunk identifier
    'document_id': str,        # Parent document ID
    'content': str,            # Full chunk text
    'chunk_index': int,        # Position in document
    'start_char': int,         # Character offset start
    'end_char': int,           # Character offset end
    'char_count': int,         # Length
    'source': str,             # File path
    'created_at': str          # ISO timestamp
}
```

## Common Modification Patterns

### Adding a New Document Format

1. **Update config.yaml** → Add extension to `supported_extensions`
2. **Add parser method** to `DocumentProcessor`:
   ```python
   def _read_newformat_file(self, file_path: Path) -> str:
       # Parse file and return text
       pass
   ```
3. **Update `_extract_text`** method:
   ```python
   elif suffix == '.newext':
       return self._read_newformat_file(file_path)
   ```

### Adding a New API Endpoint

1. **Define Pydantic models** in `app/api/models.py`
2. **Add endpoint** in `app/api/endpoints.py`:
   ```python
   @app.post("/new-endpoint", response_model=ResponseModel)
   async def new_endpoint(request: RequestModel):
       try:
           result = rag_pipeline.new_operation(request.param)
           return ResponseModel(**result)
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))
   ```

### Modifying Search Logic

**Location**: `app/core/retriever.py`

1. **Add new search mode**:
   ```python
   def _new_search_mode(self, query: str, top_k: int) -> List[SearchResult]:
       # Implementation
       pass
   ```
2. **Update `search()` method**:
   ```python
   elif mode == 'new_mode':
       results = self._new_search_mode(query, k, filter_dict)
   ```

### Changing the LLM Prompt

**Location**: `app/core/ollama_client.py`

Modify `_get_rag_template()` method:
```python
def _get_rag_template(self) -> str:
    return """Your new template here
    
    {context}
    {question}
    """
```

### Adding UI Features

**Location**: `app/ui/streamlit_app.py`

1. **Add to appropriate page function** (`render_chat_page`, `render_documents_page`, etc.)
2. **Use Streamlit components**: `st.button()`, `st.text_input()`, etc.
3. **Update session state** if needed
4. **Call backend via** `rag_pipeline` methods

## Testing Considerations

### Manual Testing Workflow

1. **Start Ollama**: `ollama serve`
2. **Pull model**: `ollama pull llama3.2:3b`
3. **Run setup**: `python -m venv venv && pip install -r requirements.txt`
4. **Start API**: `python run_api.py` (port 8000)
5. **Start UI**: `python run_ui.py` (port 8501)
6. **Upload test documents**
7. **Verify indexing** via stats
8. **Test search modes**
9. **Test chat with sources**

### Component Testing

**Embedding Engine**
```python
from app.core.embedding_engine import embedding_engine
embeddings = embedding_engine.embed_texts(["test text"])
assert embeddings.shape == (1, 384)
```

**Vector Store**
```python
from app.core.vector_store import vector_store
stats = vector_store.get_stats()
assert 'total_documents' in stats
```

**Retriever**
```python
from app.core.retriever import retriever
results = retriever.search("test query", search_mode="hybrid", top_k=5)
assert len(results) <= 5
```

## Environment & Dependencies

### Python Version
- **Minimum**: Python 3.8
- **Recommended**: Python 3.10+
- **Maximum tested**: Python 3.11

### External Dependencies
- **Ollama server** running on `http://localhost:11434`
- **Model downloaded**: Default `llama3.2:3b` or any compatible model

### Installation Commands
```bash
# Virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Dependencies
pip install -r requirements.txt

# Download embedding model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Runtime Requirements
- **CPU**: Multi-core recommended for embedding generation
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: ~2GB for models + document storage
- **Network**: Not required after initial setup (offline-capable)

## Performance Considerations

### Optimization Tips

1. **Batch Processing**
   - Use `batch_size` in config for embeddings
   - Process multiple documents together when possible

2. **Index Management**
   - Refresh BM25 index only after bulk operations
   - FAISS persists automatically on modifications

3. **Search Optimization**
   - Adjust `top_k` based on use case
   - Use semantic search for conceptual queries
   - Use keyword search for specific terms
   - Hybrid mode balances both

4. **Chunking Strategy**
   - Smaller chunks (256-512): Better precision, more chunks
   - Larger chunks (1024+): Better context, fewer chunks
   - Overlap (50-100): Prevents boundary loss

5. **Reranking**
   - Enable for better accuracy (adds latency)
   - Disable for faster responses
   - Most beneficial with hybrid search

### Memory Management

- Embedding model: ~100MB
- FAISS index: ~4 bytes per dimension per vector
- Cross-encoder (reranking): ~200MB
- Ollama model: Varies (3B model ~2GB, 7B ~4GB, etc.)

## Security & Privacy

### Data Handling
- All data stays local (no external API calls except Ollama)
- Documents stored unencrypted in `data/documents/`
- Embeddings stored in FAISS index
- Metadata in SQLite (unencrypted)

### Access Control
- No built-in authentication
- Intended for local/trusted network use
- API runs on localhost by default
- UI accessible only via localhost

### Recommendations for Production
- Add authentication middleware to FastAPI
- Use HTTPS with reverse proxy
- Encrypt data directory
- Implement rate limiting
- Add input validation/sanitization
- Restrict file upload types strictly

## Troubleshooting Guide

### Recent Fixes & Improvements

**Version: Current (2026-02-07)**

This section documents issues that have been identified and resolved in the current codebase:

#### ✅ **Fixed Issues:**

1. **Dependency Compatibility** 
   - Issue: `ImportError: cannot import name 'cached_download'`
   - Fix: Updated requirements.txt with compatible versions
   - Files: `requirements.txt` (sentence-transformers==2.3.1, huggingface-hub>=0.19.0,<0.21.0)

2. **Module Import Errors**
   - Issue: `ModuleNotFoundError: No module named 'app'` in Streamlit
   - Fix: Added automatic PYTHONPATH injection in app/ui/streamlit_app.py
   - Files: `app/ui/streamlit_app.py`, `run_ui.py`, `run_ui.bat`, `run_both.bat`

3. **Streamlit Widget Restrictions**
   - Issue: `st.chat_input() can't be used inside st.tabs`
   - Fix: Replaced st.tabs() with button-based navigation using session state
   - Files: `app/ui/streamlit_app.py`

4. **Duplicate Query Processing**
   - Issue: Queries running twice due to Streamlit reruns
   - Fix: Proper session state management, conditional reruns
   - Files: `app/ui/streamlit_app.py`

5. **Ollama Connectivity Check**
   - Issue: False negatives when checking Ollama status
   - Fix: Separated connection check from model availability check
   - Files: `app/core/ollama_client.py`, `app/ui/streamlit_app.py`

6. **Torch Warnings**
   - Issue: Annoying "examining path of torch.classes" warnings
   - Fix: Added warning suppression for torch module
   - Files: `app/ui/streamlit_app.py`

#### ✨ **New Features:**

1. **Dynamic Model Selection**
   - Feature: Select any available Ollama model from UI dropdown
   - No need to match config.yaml model exactly
   - Real-time model switching without restart
   - Files: `app/ui/streamlit_app.py`, `app/core/ollama_client.py`

2. **Enhanced Status Display**
   - Shows detailed Ollama connection status
   - Lists available models when configured model is missing
   - Clear instructions for next steps
   - Files: `app/ui/streamlit_app.py`

3. **Improved Error Messages**
   - More specific error messages for common issues
   - Actionable suggestions in UI
   - Better logging for debugging
   - Files: Multiple

### Common Issues & Solutions

#### 1. **Dependency & Import Errors**

**Issue**: `ImportError: cannot import name 'cached_download' from 'huggingface_hub'`
- **Cause**: Incompatible versions of sentence-transformers and huggingface-hub
- **Solution**: 
  ```bash
  pip uninstall sentence-transformers huggingface-hub -y
  pip install -r requirements.txt
  ```
- **Fixed in**: requirements.txt v2 (sentence-transformers==2.3.1, huggingface-hub>=0.19.0,<0.21.0)

**Issue**: `ModuleNotFoundError: No module named 'app'`
- **Cause**: Python path doesn't include project root when running Streamlit
- **Solution**: Already fixed in code with path injection:
  ```python
  # In app/ui/streamlit_app.py
  project_root = Path(__file__).parent.parent.parent
  sys.path.insert(0, str(project_root))
  ```
- **Alternative**: Ensure PYTHONPATH is set:
  ```bash
  # Windows
  set PYTHONPATH=%CD%
  
  # Linux/Mac
  export PYTHONPATH=$(pwd)
  ```

**Issue**: `StreamlitAPIException: st.chat_input() can't be used inside st.tabs`
- **Cause**: Streamlit restriction on widget placement
- **Solution**: Already fixed - using button-based navigation instead of st.tabs()
- **Implementation**: Session state tracking with manual tab switching

#### 2. **Ollama Connection Issues**

**Issue**: "❌ Ollama Not Available" despite Ollama running
- **Cause**: Could be connection check failing or model not available
- **Diagnosis Steps**:
  1. Check if Ollama is running: Visit `http://localhost:11434` in browser
  2. Check available models: `ollama list`
  3. Check logs in Streamlit sidebar for specific error
- **Solutions**:
  - Ensure `ollama serve` is running
  - Verify `ollama.base_url` in config.yaml matches your Ollama URL
  - Pull a model: `ollama pull llama3.2:3b` (or any model)

**Issue**: "⚠️ Ollama Connected but Model Missing"
- **Cause**: Ollama is running but the model specified in config isn't pulled
- **Solution**: Either:
  1. Pull the configured model: `ollama pull llama3.2:3b`
  2. Or use dynamic model selection: Select any available model from dropdown in sidebar
- **Feature**: UI now supports selecting any available model dynamically

**Issue**: Model not found
- **Diagnosis**: Run `ollama list` to see available models
- **Solution**: Pull model with `ollama pull <model-name>`
- **Popular models**:
  ```bash
  ollama pull llama3.2:3b      # Fast, 3B parameters
  ollama pull llama3.2:1b      # Fastest, 1B parameters  
  ollama pull llama2:7b        # Balanced, 7B parameters
  ollama pull mistral          # Alternative, good quality
  ```

#### 3. **Performance Issues**

**Issue**: Slow response times (>30 seconds)
- **Causes**: Large model, many chunks, reranking enabled
- **Solutions**:
  1. Use smaller model (1B or 3B instead of 7B+)
  2. Reduce `top_k` in sidebar (try 5 instead of 10)
  3. Disable reranking in config.yaml: `enable_reranking: false`
  4. Reduce chunk size: `chunking.size: 256` in config.yaml
  5. Switch to faster search mode: Use "keyword" instead of "hybrid"

**Issue**: Duplicate query processing / queries running twice
- **Cause**: Streamlit reruns triggering re-execution
- **Solution**: Already fixed in code:
  - Chat history stored in session state
  - Queries only processed on new chat input
  - Tab switching and model changes don't re-trigger queries
  - Proper use of `st.rerun()` with state checks

**Issue**: Out of memory during indexing
- **Solutions**:
  - Reduce `embedding.batch_size` in config (try 16 or 8)
  - Process fewer documents at once
  - Use smaller model
  - Increase system RAM

#### 4. **Search & Retrieval Issues**

**Issue**: No search results returned
- **Diagnosis Steps**:
  1. Check if documents are indexed: Go to "Documents" tab
  2. Verify chunk count > 0
  3. Check `min_score_threshold` in config (try lowering to 0.3)
- **Solutions**:
  - Ensure documents are uploaded and indexed
  - Try different search mode (hybrid usually best)
  - Lower score threshold in config.yaml
  - Re-index documents if corrupted

**Issue**: Irrelevant search results
- **Solutions**:
  1. Increase `min_score_threshold` to 0.6 or 0.7
  2. Enable reranking: `enable_reranking: true`
  3. Try semantic search mode for conceptual queries
  4. Try keyword mode for specific term searches
  5. Adjust `hybrid_alpha` (higher = more semantic)

**Issue**: FAISS index corruption
- **Symptoms**: Errors loading index, dimension mismatches
- **Solution**: 
  ```bash
  # Delete and rebuild indices
  rm -rf data/vector_indices/
  # Re-upload documents through UI
  ```

#### 5. **Database Issues**

**Issue**: SQLite database locked
- **Cause**: Concurrent write attempts
- **Solutions**:
  - Close other instances of the app
  - Check file permissions on `data/metadata.db`
  - Delete and recreate: `rm data/metadata.db` (will require re-indexing)

**Issue**: Metadata inconsistency
- **Symptoms**: Document count mismatch, missing chunks
- **Solution**: Clear and re-index:
  ```bash
  rm -rf data/vector_indices/
  rm data/metadata.db
  # Re-upload documents
  ```

#### 6. **File Upload Issues**

**Issue**: File upload fails
- **Solutions**:
  - Check file size < `max_file_size_mb` (default 50MB)
  - Verify file extension in `supported_extensions` list
  - Check disk space in `data/documents/`
  - Ensure write permissions on data directory

**Issue**: Embedding model download fails
- **Cause**: No internet connection or firewall blocking
- **Solutions**:
  - Check internet connection
  - Manually download to `models/` directory
  - Configure proxy if needed
  - Use pre-downloaded models

#### 7. **UI Issues**

**Issue**: Streamlit warnings about torch
- **Message**: "Examining the path of torch.classes raised..."
- **Cause**: PyTorch checking for optional CUDA extensions
- **Impact**: None - just a warning
- **Solution**: Already suppressed in code:
  ```python
  warnings.filterwarnings('ignore', category=UserWarning, module='torch')
  logging.getLogger('torch').setLevel(logging.ERROR)
  ```

**Issue**: UI not loading / blank page
- **Solutions**:
  1. Check console for errors
  2. Verify virtual environment is activated
  3. Reinstall dependencies: `pip install -r requirements.txt`
  4. Clear Streamlit cache: `streamlit cache clear`
  5. Try different browser

**Issue**: Chat history lost
- **Cause**: Session ended or page refreshed
- **Behavior**: This is normal - Streamlit session state is temporary
- **Future Enhancement**: Could add persistent storage

#### 8. **API Issues**

**Issue**: API not starting / port already in use
- **Error**: "Address already in use" on port 8000
- **Solutions**:
  ```bash
  # Windows: Find and kill process
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F
  
  # Linux/Mac
  lsof -ti:8000 | xargs kill -9
  
  # Or use different port in run_api.py
  uvicorn.run(app, host="0.0.0.0", port=8001)
  ```

**Issue**: API endpoint returns 500 error
- **Diagnosis**: Check API logs for stack trace
- **Common causes**:
  - Ollama not running
  - Model not available
  - Document not indexed
  - Invalid request format

### Diagnostic Commands

```bash
# Check Python environment
python --version
pip list | grep -E "streamlit|fastapi|sentence-transformers"

# Check Ollama
curl http://localhost:11434/api/tags
ollama list
ollama ps  # Show running models

# Check files and permissions
ls -la data/
ls -la data/vector_indices/
ls -la data/documents/

# Test imports
python -c "from app.core.config import config; print(config.ollama_model)"
python -c "from app.core.rag_pipeline import rag_pipeline; print(rag_pipeline.get_stats())"

# Check ports
netstat -an | grep -E "8000|8501|11434"  # Linux/Mac
netstat -an | findstr "8000 8501 11434"   # Windows
```

### Performance Benchmarks

Expected performance on modern hardware:

| Operation | Time | Notes |
|-----------|------|-------|
| Single query embedding | 0.1-0.3s | CPU-based |
| FAISS search (10K chunks) | 0.05-0.1s | Very fast |
| BM25 search | 0.1-0.2s | Python-based |
| Reranking (10 results) | 0.3-0.5s | Cross-encoder |
| LLM generation (3B model) | 5-15s | Depends on length |
| LLM generation (7B model) | 15-30s | Slower but better |
| Document indexing (1MB) | 2-5s | Including chunking |
| Batch embedding (32 chunks) | 1-2s | Batched processing |

### Logging & Debugging

**Enable detailed logging:**

```python
# Add to top of any module for debugging
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Key log locations:**
- Streamlit console output (where you ran `python run_ui.py`)
- API console output (where you ran `python run_api.py`)
- Ollama logs (Ollama console)

**Important log messages:**
```
INFO:app.core.rag_pipeline:Processing query: [question]
Batches: 100%|████████| 1/1 [00:00<00:00, 3.33it/s]  # Normal embedding progress
INFO:app.core.retriever:Retrieved X results in hybrid mode
INFO:app.core.ollama_client:Generating response...
```

### Known Limitations

1. **No built-in authentication** - Runs on localhost only
2. **Session state is temporary** - Chat history lost on refresh
3. **Single-user design** - Not optimized for concurrent users
4. **CPU-only by default** - Can be slow for large documents (GPU support possible)
5. **No document versioning** - Overwrites on re-upload
6. **Limited file size** - Default 50MB per file
7. **No incremental indexing** - Full re-index on document changes

## Version Control & Collaboration

### Files to Ignore (.gitignore suggestions)
```
venv/
__pycache__/
*.pyc
.env
data/
models/
*.db
*.bin
*.pkl
.streamlit/
```

### Important Files to Commit
- All `.py` files
- `config.yaml` (without secrets)
- `requirements.txt`
- `README.md`, `AGENT.md`
- `.bat` files for Windows users

## Extension Points

### Planned/Possible Extensions

1. **Authentication Layer**
   - FastAPI middleware
   - User management
   - Document-level permissions

2. **Cloud Storage Integration**
   - S3/Azure Blob for documents
   - Remote vector databases

3. **Advanced Analytics**
   - Query logging
   - Usage statistics
   - Performance metrics

4. **Multi-Modal Support**
   - Image documents (OCR)
   - Audio transcription

5. **Collaborative Features**
   - Shared knowledge bases
   - Multi-user chat

6. **Enhanced UI**
   - Document preview
   - Annotation tools
   - Export functionality

## UI Features & User Guide

### Search Modes Explained

The system offers three search modes for document retrieval:

#### 🎯 **Semantic Search**
- **How it works**: Uses AI embeddings to understand the meaning of your query
- **Strength**: Finds conceptually similar content even without exact word matches
- **Best for**: 
  - Conceptual questions
  - Finding related ideas
  - When you don't know exact terminology
- **Example**: Query "improve performance" finds "optimization", "speed enhancement", "efficiency improvements"

#### 🔤 **Keyword Search (BM25)**
- **How it works**: Traditional keyword matching with statistical relevance scoring
- **Strength**: Precise matching of specific terms
- **Best for**:
  - Finding specific terms, names, codes
  - Technical keyword searches
  - Exact phrase matching
- **Example**: Query "error code 404" finds exact mentions of "404"

#### ⚖️ **Hybrid (Recommended Default)**
- **How it works**: Combines semantic (70%) and keyword (30%) search
- **Strength**: Best of both worlds - meaning + precision
- **Best for**: Most use cases
- **Configuration**: `hybrid_alpha` in config.yaml (default 0.7)
  - Higher values = more semantic
  - Lower values = more keyword-based
- **Example**: Query "database connection issues" finds both "DB connectivity problems" (semantic) and exact "database connection" mentions (keyword)

### Results to Retrieve (Top-K)

Controls how many document chunks are sent to the LLM as context.

#### **Default: 10 chunks**
- Each chunk typically ~512 characters
- Provides balanced context for most queries

#### **Trade-offs:**

| Top-K Value | Speed | Coverage | Best For |
|-------------|-------|----------|----------|
| 3-5 | ⚡ Fastest | Limited | Simple queries, quick answers |
| 10-15 | ⚖️ Balanced | Good | Most use cases (default) |
| 20-30 | 🐌 Slower | Comprehensive | Complex research questions |

#### **With Re-ranking (enabled by default):**
1. Retrieves Top-K chunks (e.g., 10)
2. Re-scores using cross-encoder model
3. Returns top `rerank_top_k` (default: 5) most relevant
4. More accurate but adds ~0.5s processing time

### Dynamic Model Selection

The UI now supports selecting any available Ollama model dynamically:

#### **Features:**
- Automatic detection of all installed Ollama models
- Dropdown selector in sidebar
- Real-time model switching
- No restart required
- Defaults to model in config.yaml if available

#### **How to use:**
1. Ensure Ollama is running: `ollama serve`
2. Pull any models you want: `ollama pull llama3.2:3b`
3. Open the UI - available models appear in dropdown
4. Select your preferred model
5. Start chatting - responses use selected model

#### **Model Recommendations:**

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| llama3.2:1b | ~1GB | ⚡⚡⚡ Fast | Good | Quick answers, testing |
| llama3.2:3b | ~2GB | ⚡⚡ Fast | Better | General use, balance |
| llama2:7b | ~4GB | ⚡ Moderate | Great | Quality answers |
| mistral | ~4GB | ⚡ Moderate | Great | Alternative, good reasoning |
| llama2:13b | ~8GB | 🐌 Slow | Excellent | Best quality, if you have RAM |

### UI Navigation

#### **Tab-Based Interface:**
- **💬 Chat**: Ask questions, get AI-powered answers with sources
- **📁 Documents**: Upload, view, delete documents
- **🔍 Search**: Test search modes without LLM generation

#### **Sidebar Controls:**
- **Ollama Status**: Shows connection and model availability
- **Model Selection**: Choose from available models
- **Search Mode**: Select semantic/keyword/hybrid
- **Results to Retrieve**: Adjust Top-K (1-20)

#### **Chat Features:**
- Session-based chat history (lost on refresh)
- Source citations with relevance scores
- Expandable source viewer
- Current model indicator
- Clear error messages

### Normal System Behavior

Understanding what's expected vs. what's an issue:

#### ✅ **Normal Behaviors:**

1. **First-time startup is slow** (~30s)
   - Loading embedding model
   - Initializing FAISS index
   - Normal and only happens once

2. **Query processing shows progress**
   ```
   INFO:app.core.rag_pipeline:Processing query: [your question]
   Batches: 100%|████████| 1/1 [00:00<00:00, 3.33it/s]
   ```
   - This is expected for every query
   - Shows embedding generation progress

3. **Response time varies** (5-30 seconds)
   - Depends on model size
   - Depends on answer length
   - Depends on Top-K setting
   - Completely normal

4. **Torch warnings** (now suppressed in code)
   - Were checking for CUDA extensions
   - Harmless, ignored by default now

5. **Chat history cleared on refresh**
   - Streamlit session state is temporary
   - Design limitation, not a bug
   - Future enhancement: persistent storage

#### ⚠️ **Abnormal Behaviors:**

1. **Queries processing twice** - FIXED
   - Was a rerun issue
   - Now resolved with proper state management

2. **No results despite having documents**
   - Check if documents are indexed
   - Try lowering `min_score_threshold`
   - Re-index if needed

3. **Errors on every query**
   - Check Ollama is running
   - Check model is available
   - Check logs for specific error

## Quick Reference

### Key Files for Common Tasks

| Task | Primary Files |
|------|---------------|
| Add document format | `document_processor.py`, `config.yaml` |
| Modify search | `retriever.py` |
| Change LLM behavior | `ollama_client.py` |
| Add API endpoint | `endpoints.py`, `models.py` |
| Update UI | `streamlit_app.py` |
| Adjust chunking | `document_processor.py`, `config.yaml` |
| Database schema | `vector_store.py` |
| Configuration | `config.yaml`, `config.py` |

### Useful Commands

```bash
# Start services
python run_api.py              # API on :8000
python run_ui.py               # UI on :8501

# Ollama management
ollama serve                   # Start Ollama
ollama pull llama3.2:3b       # Download model
ollama list                    # List models

# Development
pip install -r requirements.txt
python -m pytest              # If tests exist
python -m black .             # Code formatting (optional)
python -m mypy app/           # Type checking (optional)
```

### Configuration Defaults

| Setting | Default | Purpose |
|---------|---------|---------|
| Embedding Model | all-MiniLM-L6-v2 | Text embeddings |
| LLM Model | llama3.2:3b | Question answering |
| Search Mode | hybrid | Retrieval strategy |
| Chunk Size | 512 | Characters per chunk |
| Chunk Overlap | 50 | Overlap chars |
| Top-K | 10 | Results to retrieve |
| Hybrid Alpha | 0.7 | Semantic weight |
| Temperature | 0.7 | LLM randomness |

---

## Summary for AI Coding Agents

When working on this codebase:

1. **Follow existing patterns**: Singleton services, dataclasses, type hints
2. **Use configuration**: Never hardcode values that belong in `config.yaml`
3. **Log extensively**: Use module-level loggers for all operations
4. **Handle errors**: Try-except with logging and appropriate re-raising
5. **Type everything**: All functions need type hints
6. **Test locally**: Requires Ollama running with a model
7. **Document changes**: Update this file when adding major features
8. **Maintain offline capability**: No external API dependencies
9. **Preserve modularity**: Each component should work independently
10. **Think security**: This handles user documents - be cautious

**Primary Design Principle**: Offline-first, modular, production-ready RAG system for secure environments.
