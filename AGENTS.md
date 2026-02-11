# AGENTS.md - Coding Agent Instructions

## Build/Lint/Test Commands

### Setup
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Application
```bash
# Run API server only
python main.py api

# Run UI only  
python main.py ui

# Run both (Windows)
run_both.bat
```

### Testing
```bash
# Run all tests
python -m pytest

# Run single test file
python -m pytest tests/test_file.py

# Run single test function
python -m pytest tests/test_file.py::test_function_name

# Run with coverage
python -m pytest --cov=app tests/
```

### Linting & Formatting
```bash
# Format code with black
python -m black app/ tests/

# Check with flake8
python -m flake8 app/ tests/

# Type checking with mypy
python -m mypy app/

# Sort imports
python -m isort app/ tests/
```

## Code Style Guidelines

### Import Order
1. Standard library imports (alphabetical)
2. Third-party imports (alphabetical)
3. Local imports (alphabetical)

```python
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import yaml
from pydantic import BaseModel

from app.core.config import config
```

### Type Hints
- All function parameters and return types must be typed
- Use `typing` module: `List`, `Dict`, `Optional`, `Any`, `Tuple`

```python
def process_file(self, file_path: str, document_id: Optional[str] = None) -> ProcessedDocument:
    pass
```

### Naming Conventions
- Classes: `PascalCase` (e.g., `VectorStore`, `DocumentProcessor`)
- Functions/variables: `snake_case` (e.g., `get_stats`, `chunk_size`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_FILE_SIZE`)
- Private methods: `_leading_underscore` (e.g., `_load_config`)
- Module-level singletons: lowercase (e.g., `config`, `vector_store`)

### Class Structure
```python
class VectorStore:
    def __init__(self, index_path: Optional[str] = None):
        self.index_path = Path(index_path or config.vector_index_path)
        self._init_metadata_db()
    
    def _init_metadata_db(self):
        """Private helper method."""
        pass
    
    def add_chunks(self, chunks: List[Dict]) -> None:
        """Public API method."""
        pass
```

### Dataclasses
Use `@dataclass` for data containers instead of dictionaries:

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    document_id: str
```

### Logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Processing: {file_path}")
logger.error(f"Error: {str(e)}")
```

### Error Handling
- Use specific exceptions when possible
- Log errors before raising
- Re-raise with context when appropriate

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

### Configuration Access
Use the `config` singleton from `app.core.config`:

```python
from app.core.config import config

# Via properties
model = config.ollama_model
chunk_size = config.chunk_size

# Via get method with nested keys and default
timeout = config.get('ollama.timeout', 120)
```

### File Paths
Use `pathlib.Path` for all path operations:

```python
from pathlib import Path

file_path = Path(config.documents_path) / filename
file_path.parent.mkdir(parents=True, exist_ok=True)
```

### String Formatting
- Use f-strings for interpolation
- Multi-line strings with triple quotes for prompts/SQL

```python
# f-strings
logger.info(f"Processing {filename}")

# Multi-line
query = """
    SELECT * FROM documents 
    WHERE id = ?
"""
```

### Database (SQLite)
- Use parameterized queries (never string interpolation)
- Connection per operation with context managers

```python
conn = sqlite3.connect(self.metadata_db_path)
cursor = conn.cursor()
cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
conn.commit()
conn.close()
```

### FastAPI Patterns
- Use Pydantic models for request/response
- Async handlers for I/O operations
- HTTP status codes via `HTTPException`

```python
@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        result = process_file(file)
        return DocumentUploadResponse(success=True, **result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Key Patterns

### Singleton Pattern
Module-level instantiation for shared components:

```python
# At end of module
vector_store = VectorStore()
```

### Properties for Config
Use `@property` decorators for configuration values:

```python
@property
def ollama_model(self) -> str:
    return self.get('ollama.model', 'llama3.2:3b')
```

## Project Structure

```
ILIN/
├── app/
│   ├── api/              # FastAPI endpoints
│   ├── core/             # Business logic
│   │   ├── llm/          # LLM clients
│   │   ├── retrieval/    # Search & embeddings
│   │   └── indexing/     # Document processing
│   ├── scripts/          # Entry points
│   └── ui/               # Streamlit UI
├── config.yaml           # Configuration
├── requirements.txt      # Dependencies
└── main.py              # Launcher
```

## Critical Notes

- **Never hardcode values** - use `config.yaml`
- **Always use type hints** - all functions need them
- **Log extensively** - use module-level loggers
- **Handle errors** - try-except with logging
- **Maintain offline capability** - no external APIs except Ollama
- **Use pathlib** - never string paths
- **Preserve modularity** - components work independently
