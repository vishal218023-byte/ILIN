import sys
from pathlib import Path
import warnings
import logging

# Add project root to Python path for imports to work
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Suppress torch warnings and inspection errors
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
logging.getLogger('torch').setLevel(logging.ERROR)

# Suppress specific "torch.classes" inspection errors
class TorchInspectionFilter(logging.Filter):
    def filter(self, record):
        return "torch.classes" not in record.getMessage()

logging.getLogger().addFilter(TorchInspectionFilter())


import streamlit as st
import requests
import json
import time

from app.core.config import config
from app.core.rag_pipeline import rag_pipeline
# Using OllamaClient from llm_client

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title=config.get('ui.page_title', 'ILIN - Intelligence Node'),
    page_icon=config.get('ui.page_icon', '🧠'),
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: rgba(31, 119, 180, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #1f77b4;
        color: inherit;
    }
    .score-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .result-card {
        background-color: rgba(151, 166, 195, 0.05);
        border: 1px solid rgba(151, 166, 195, 0.2);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'engine_ready' not in st.session_state:
        st.session_state.engine_ready = False
    
    # Persistent settings from config
    if 'engine_type' not in st.session_state:
        st.session_state.engine_type = config.get('last_engine', "Ollama")
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = config.get('online_api.groq.api_key', '')
    if 'selected_model_ollama' not in st.session_state:
        st.session_state.selected_model_ollama = config.get('ollama.model', 'llama3.2:3b')
    if 'selected_model_gguf' not in st.session_state:
        st.session_state.selected_model_gguf = config.get('local_llm.last_model', None)
    if 'selected_model_groq' not in st.session_state:
        st.session_state.selected_model_groq = config.get('online_api.groq.default_model', 'llama-3.3-70b-versatile')

@st.cache_data(ttl=60, show_spinner="Connecting...")
def check_ollama_status():
    """Check if Ollama is running and model is available. Cached for 60 seconds."""
    from app.core.llm.llm_client import OllamaClient
    client = OllamaClient()
    try:
        # First check if Ollama is running
        if not client.check_connection():
            return {'connected': False, 'model_available': False, 'models': []}
        
        # Then check if model is available
        available_models = client.list_models()
        model_available = client.check_model_available()
        
        return {
            'connected': True,
            'model_available': model_available,
            'models': available_models
        }
    except Exception as e:
        return {'connected': False, 'model_available': False, 'models': [], 'error': str(e)}


@st.cache_data(ttl=3600, show_spinner="Fetching Groq models...")
def get_groq_models(api_key):
    """Fetch Groq models and cache for 1 hour."""
    from app.core.llm.llm_client import GroqClient
    try:
        temp_client = GroqClient(api_key=api_key, model="llama-3.3-70b-versatile")
        return temp_client.list_models()
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]

def render_sidebar():
    with st.sidebar:
        st.markdown("<div class='main-header'>🧠 ILIN</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Integrated Localized Intelligence Node</div>", unsafe_allow_html=True)
        
        # 1. Engine Selection
        st.markdown("### 🔌 LLM Engine")
        engine_type = st.selectbox(
            "Select Engine",
            ["Ollama", "Local GGUF", "Online API (Groq)"],
            key="engine_type",
            help="Choose the backend that powers the AI responses."
        )
        
        # Save engine type to config if it changed
        if st.session_state.engine_type != config.get('last_engine'):
            config.set('last_engine', st.session_state.engine_type)
            config.save()
        
        # 2. Dynamic Settings based on Engine
        selected_model = None
        
        if engine_type == "Ollama":
            ollama_status = check_ollama_status()
            available_models = ollama_status.get('models', [])
            if ollama_status.get('connected', False):
                if available_models:
                    st.success(f"Engine Connected {len(available_models)} Models")
                    
                    # Ensure current selection is in available models
                    if st.session_state.selected_model_ollama not in available_models:
                        st.session_state.selected_model_ollama = available_models[0]
                    
                    selected_model = st.selectbox(
                        "Select Model", 
                        available_models, 
                        key="selected_model_ollama"
                    )
                    
                    # Persistence
                    if selected_model != config.get('ollama.model'):
                        config.set('ollama.model', selected_model)
                        config.save()
                else:
                    st.warning("⚠️ Ollama Connected but No Models Found")
            else:
                st.error("❌ Ollama Not Available")

        elif engine_type == "Local GGUF":
            models_dir = Path("models")
            available_models = [f.name for f in models_dir.glob("*.gguf")]
            if available_models:
                # Ensure current selection is in available models
                if st.session_state.selected_model_gguf not in available_models:
                    st.session_state.selected_model_gguf = available_models[0]
                    
                selected_model = st.selectbox(
                    "Select GGUF Model", 
                    available_models, 
                    key="selected_model_gguf"
                )
                
                # Persistence
                if selected_model != config.get('local_llm.last_model'):
                    config.set('local_llm.last_model', selected_model)
                    config.save()
                    
                full_model_path = str(models_dir / selected_model)
            else:
                st.error("❌ No GGUF models found in models/ directory.")

        elif engine_type == "Online API (Groq)":
            st.markdown("#### Groq Settings")
            api_key = st.text_input("Groq API Key", type="password", key="groq_api_key")
            
            # Save API key to config immediately when changed
            if api_key != config.get('online_api.groq.api_key'):
                config.set('online_api.groq.api_key', api_key)
                config.save()
            
            if api_key:
                available_models = get_groq_models(api_key)
                
                # Check if current st.session_state value is valid
                if st.session_state.selected_model_groq not in available_models:
                    st.session_state.selected_model_groq = available_models[0]
                    
                selected_model = st.selectbox(
                    "Select Model", 
                    available_models, 
                    key="selected_model_groq"
                )
                
                # Persistence
                if selected_model != config.get('online_api.groq.default_model'):
                    config.set('online_api.groq.default_model', selected_model)
                    config.save()
            else:
                st.warning("⚠️ Please enter your Groq API Key.")

        # 3. Apply Engine Settings to Global Client
        from app.core.llm.llm_client import set_llm_client
        try:
            if engine_type == "Ollama" and selected_model:
                set_llm_client("Ollama")
                from app.core.llm.llm_client import get_llm_client
                get_llm_client().model = selected_model
                st.session_state.engine_ready = True
            elif engine_type == "Local GGUF" and selected_model:
                set_llm_client("Local GGUF", model_path=full_model_path)
                st.session_state.engine_ready = True
            elif engine_type == "Online API (Groq)" and selected_model and api_key:
                set_llm_client("Online API (Groq)", api_key=api_key, model=selected_model)
                st.session_state.engine_ready = True
            else:
                st.session_state.engine_ready = False
        except Exception as e:
            st.error(f"Failed to initialize engine: {str(e)}")
            st.session_state.engine_ready = False
        
        
        st.divider()
        use_rag = st.toggle(
            "System Knowledge",
            value=True,
            help="Toggle ON for System related answers, OFF for direct LLM chat."
        )
        
        st.divider()
        st.markdown("### ⚙️ Settings")
        
        search_mode = st.selectbox(
            "Default Search Mode",
            ["hybrid", "semantic", "keyword"],
            index=0
        )
        
        top_k = st.slider("Results to Retrieve", 5, 50, 10)
        
        if st.button("🔄 Refresh Index", use_container_width=True):
            with st.spinner("Refreshing..."):
                rag_pipeline.retriever.refresh_index()
                st.cache_data.clear()
            st.success("Index refreshed!")
        
        st.divider()
        st.caption("Developed by - Vishal Raj V, E218023")
        
        return search_mode, top_k, use_rag

def render_chat_page(search_mode, top_k, use_rag):
    st.markdown("<div class='main-header'>💬 Chat with Your Documents</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Ask questions and get answers based on your knowledge base</div>", unsafe_allow_html=True)
    
    if not st.session_state.engine_ready:
        st.warning("⚠️ LLM Engine is not ready. Please check the sidebar to select a model or engine.")
        return
    
    # Display current model being used
    current_model = None
    if st.session_state.engine_type == "Ollama":
        current_model = st.session_state.selected_model_ollama
    elif st.session_state.engine_type == "Local GGUF":
        current_model = st.session_state.selected_model_gguf
    elif st.session_state.engine_type == "Online API (Groq)":
        current_model = st.session_state.selected_model_groq
        
    if current_model:
        st.caption(f"💡 Using engine: **{st.session_state.engine_type}** | Model: **{current_model}**")
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
                
                if message['role'] == 'assistant' and 'sources' in message:
                    with st.expander("📚 View Sources"):
                        for source in message['sources']:
                            st.markdown(f"""
                            <div class='source-box'>
                                <strong>{source['filename']}</strong> 
                                <span class='score-badge'>Score: {source['relevance_score']:.3f}</span>
                                <br><small>Type: {source['search_type']}</small>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Use a unique key for chat input to prevent duplicate processing
    if prompt := st.chat_input("Ask a question about your documents...", key="chat_input"):
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': prompt
        })
        
        # Generate response
        with st.spinner("Searching and generating answer..."):
            try:
                response = rag_pipeline.query(
                    question=prompt,
                    search_mode=search_mode,
                    top_k=top_k,
                    stream=False,
                    use_rag=use_rag
                )

                
                answer = response['answer']
                sources = response['sources']
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': answer,
                    'sources': sources
                })
                
            except Exception as e:
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': f"Error: {str(e)}",
                    'sources': []
                })
        
        # Rerun to display the new messages
        st.rerun()
    
    if st.button("🗑️ Clear Conversation", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

def render_documents_page():
    st.markdown("<div class='main-header'>📁 Document Management</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload, view, and manage your documents</div>", unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📤 Upload", "📋 Documents"])
    
    with tab1:
        st.markdown("### Upload New Documents")
        
        uploaded_files = st.file_uploader(
            "Drag and drop files here",
            type=['txt', 'pdf', 'docx', 'md', 'html', 'csv'],
            accept_multiple_files=True,
            help="Supported formats: TXT, PDF, DOCX, MD, HTML, CSV (Max 50MB per file)"
        )
        
        if uploaded_files:
            st.markdown("### Files Ready for Upload")
            
            for file in uploaded_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {file.name}")
                with col2:
                    st.write(f"{(file.size / 1024 / 1024):.2f} MB")
                with col3:
                    st.write("✅ Ready")
            
            if st.button("🚀 Process All Documents", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    try:
                        status_text.text(f"Processing {file.name}...")
                        
                        from app.core.config import config
                        from pathlib import Path
                        import shutil
                        
                        documents_path = Path(config.documents_path)
                        documents_path.mkdir(parents=True, exist_ok=True)
                        
                        file_path = documents_path / file.name
                        with open(file_path, "wb") as f:
                            f.write(file.getvalue())
                        
                        processed = rag_pipeline.ingest_document(str(file_path))
                        
                        progress = (idx + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        
                        st.cache_data.clear()
                        st.success(f"✅ {file.name}: {processed.total_chunks} chunks indexed")
                        
                    except Exception as e:
                        st.error(f"❌ {file.name}: {str(e)}")
                
                status_text.text("Processing complete!")
                time.sleep(1)
                st.rerun()
    
    with tab2:
        st.markdown("### Document Library")
        
        try:
            documents = rag_pipeline.list_documents()
            
            if not documents:
                st.info("No documents in the knowledge base yet. Upload some documents first!")
            else:
                for doc in documents:
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.write(f"📄 **{doc['filename']}**")
                        st.caption(f"ID: {doc['document_id'][:20]}...")
                    
                    with col2:
                        st.write(f"{doc['total_chunks']} chunks")
                    
                    with col3:
                        size_mb = doc['file_size'] / (1024 * 1024)
                        st.write(f"{size_mb:.2f} MB")
                    
                    with col4:
                        if st.button("🗑️ Delete", key=f"del_{doc['document_id']}"):
                            if rag_pipeline.delete_document(doc['document_id']):
                                st.cache_data.clear()
                                st.success("Deleted!")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error("Failed to delete")
                    
                    st.divider()
                    
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")

def render_search_page():
    st.markdown("<div class='main-header'>🔍 Advanced Search</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Search through your knowledge base with different modes</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Search query", placeholder="Enter your search terms...")
    
    with col2:
        search_mode = st.selectbox(
            "Search Mode",
            ["hybrid", "semantic", "keyword"],
            help="Hybrid combines semantic and keyword search"
        )
    
    top_k = st.slider("Number of results", 5, 50, 10)
    
    if query:
        if st.button("🔍 Search", type="primary", use_container_width=True):
            with st.spinner("Searching..."):
                try:
                    results = rag_pipeline.search_only(
                        query=query,
                        search_mode=search_mode,
                        top_k=top_k
                    )
                    
                    if not results:
                        st.info("No results found. Try a different query or search mode.")
                    else:
                        st.markdown(f"### Found {len(results)} results")
                        
                        for result in results:
                            with st.container():
                                st.markdown(f"""
                                <div class='result-card'>
                                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                                        <span><strong>#{result['rank']}</strong> {result['source'].split('/')[-1]}</span>
                                        <span class='score-badge'>Score: {result['score']:.4f}</span>
                                    </div>
                                    <div style='color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;'>
                                        Mode: {result['search_type']} | Chunk: {result['chunk_id'][:30]}...
                                    </div>
                                    <div style='background-color: #f8f9fa; padding: 0.75rem; border-radius: 0.3rem; font-size: 0.9rem;'>
                                        {result['content']}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
    
    st.divider()
    
    with st.expander("ℹ️ Search Mode Help"):
        st.markdown("""
        **Semantic Search**: Finds conceptually similar content using AI embeddings. Good for finding related concepts even with different keywords.
        
        **Keyword Search**: Traditional text matching using BM25 algorithm. Good for exact matches and specific terms.
        
        **Hybrid Search**: Combines both approaches with a weighted score. Best overall performance for most queries.
        """)

def main():
    init_session_state()
    
    search_mode, top_k, use_rag = render_sidebar()
    
    # Use session state to track selected tab instead of st.tabs for chat_input compatibility
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Chat"
    
    # Create tab buttons (only rerun if tab actually changes)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💬 Chat", use_container_width=True, type="primary" if st.session_state.active_tab == "Chat" else "secondary"):
            if st.session_state.active_tab != "Chat":
                st.session_state.active_tab = "Chat"
                st.rerun()
    with col2:
        if st.button("📁 Documents", use_container_width=True, type="primary" if st.session_state.active_tab == "Documents" else "secondary"):
            if st.session_state.active_tab != "Documents":
                st.session_state.active_tab = "Documents"
                st.rerun()
    with col3:
        if st.button("🔍 Search", use_container_width=True, type="primary" if st.session_state.active_tab == "Search" else "secondary"):
            if st.session_state.active_tab != "Search":
                st.session_state.active_tab = "Search"
                st.rerun()
    
    st.divider()
    
    # Render the active tab
    if st.session_state.active_tab == "Chat":
        render_chat_page(search_mode, top_k, use_rag)

    elif st.session_state.active_tab == "Documents":
        render_documents_page()
    elif st.session_state.active_tab == "Search":
        render_search_page()

if __name__ == "__main__":
    main()
