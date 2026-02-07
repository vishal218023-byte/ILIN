# ILIN - Integrated Localized Intelligence Node

An AI-based local assistance system designed to work completely offline in a secure air-gapped environment for local RAG (Retrieval-Augmented Generation) implementation.

## Features

- **Completely Offline**: Works without internet connectivity
- **Multi-Format Document Support**: PDF, DOCX, TXT, MD, HTML, CSV
- **Advanced Search**: Semantic, keyword, and hybrid search modes with re-ranking
- **Local LLM Integration**: Powered by Ollama with support for various models
- **Vector Database**: FAISS for efficient similarity search
- **Modern UI**: Streamlit-based web interface
- **REST API**: FastAPI endpoints for programmatic access

## Architecture

```
ILIN/
├── app/
│   ├── core/
│   │   ├── config.py              # Configuration management
│   │   ├── document_processor.py  # Text extraction and chunking
│   │   ├── embedding_engine.py    # Sentence transformers
│   │   ├── vector_store.py        # FAISS integration
│   │   ├── retriever.py           # Advanced search logic
│   │   ├── ollama_client.py       # LLM integration
│   │   └── rag_pipeline.py        # Main RAG orchestrator
│   ├── api/
│   │   ├── models.py              # Pydantic models
│   │   └── endpoints.py           # FastAPI endpoints
│   └── ui/
│       └── streamlit_app.py       # Web interface
├── data/
│   ├── documents/                 # Document storage
│   ├── vector_indices/            # FAISS indices
│   └── metadata.db               # SQLite metadata
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
├── run_api.py                    # Start API server
└── run_ui.py                     # Start UI
```

## Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running locally
   - Install from: https://ollama.ai
   - Pull a model: `ollama pull llama3.2:3b`

## Installation

1. **Clone/Navigate to the project directory:**
   ```bash
   cd ilin
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download embedding model** (first run only):
   ```python
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```

## Usage

### Option 1: Run Both API and UI (Recommended)

Start the API server in one terminal:
```bash
python run_api.py
```

Start the UI in another terminal:
```bash
python run_ui.py
```

Then open your browser to: `http://localhost:8501`

### Option 2: Use UI Only (Direct Mode)

```bash
python -m streamlit run app/ui/streamlit_app.py
```

### Option 3: API Only

```bash
python run_api.py
```

API will be available at: `http://localhost:8000`

## Configuration

Edit `config.yaml` to customize:

```yaml
ollama:
  model: "llama3.2:3b"  # Change to your preferred model
  
retrieval:
  search_mode: "hybrid"  # semantic, keyword, or hybrid
  
chunking:
  size: 512  # Token size per chunk
  overlap: 50  # Overlap between chunks
```

## Using the System

### 1. Document Upload
- Go to the **Documents** tab
- Click **Upload** sub-tab
- Drag & drop files or click to select
- Click **Process All Documents**
- Documents are automatically indexed

### 2. Chat with Documents
- Go to the **Chat** tab
- Type your question
- View the AI response with source citations
- Clear conversation when done (resets each session)

### 3. Advanced Search
- Go to the **Search** tab
- Enter search query
- Select search mode (semantic/keyword/hybrid)
- View ranked results with relevance scores

## API Endpoints

- `GET /` - Health check
- `POST /documents/upload` - Upload document
- `GET /documents` - List all documents
- `DELETE /documents/{id}` - Delete document
- `POST /search` - Search documents
- `POST /chat` - Chat with RAG
- `GET /stats` - Knowledge base statistics

## Supported Document Formats

- **.txt** - Plain text
- **.pdf** - PDF documents
- **.docx** - Microsoft Word
- **.md** - Markdown
- **.html** - HTML files
- **.csv** - CSV files

## Search Modes

1. **Semantic**: AI-powered similarity search (best for concepts)
2. **Keyword**: Traditional text matching (best for exact terms)
3. **Hybrid**: Combines both approaches (recommended)

## System Requirements

- **CPU**: Modern multi-core processor
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB for models + document storage
- **OS**: Windows 10/11, Linux, macOS

## Security Notes

- All data stays local
- No internet connection required after setup
- Documents stored in `data/documents/`
- Vector index stored in `data/vector_indices/`
- Metadata in SQLite database

## Troubleshooting

**Ollama not connected:**
- Ensure Ollama is running: `ollama serve`
- Check model is pulled: `ollama list`

**No results in search:**
- Verify documents are indexed (check stats)
- Try different search mode
- Check chunk size in config

**Memory issues:**
- Reduce batch size in config
- Use smaller model
- Increase chunk overlap

## License

MIT License - Local RAG System

## Credits

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: FAISS by Facebook
- **LLM**: Ollama framework
- **UI**: Streamlit
- **API**: FastAPI
