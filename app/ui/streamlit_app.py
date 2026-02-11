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

# Create module-level logger
logger = logging.getLogger(__name__)

import streamlit as st
import requests
import json
import time

from app.core.config import config
from app.core.rag_pipeline import rag_pipeline
# Using OllamaClient from llm_client

API_BASE_URL = "http://localhost:8000"

from PIL import Image

# Load Favicon
icon_path = Path(__file__).parent / "assets" / "Icon.png"
if icon_path.exists():
    favicon = Image.open(icon_path)
else:
    favicon = "🧠"

st.set_page_config(
    page_title=config.get('ui.page_title', 'ILIN - Intelligence Node'),
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base64 encoded logo for embedding
LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="auto" viewBox="0 0 1541 889">
<g fill="#00aeef">
<path d="m76.2 847.9-7 .1-6.7-.1-5.8.1h-5.4l-4.9.1-4.1.2-4 .3-3.4.2-3.1.2-2.6.2-2.3.4-2.1.5-1.8.5-1.6.7-1.3.7-1.1.6-.9 1-1 .9-.7 1-.8 1.1-14.6 23-1 3 .8 3.1 2.1 2.2 3 .7 1032.7.3 3.8-.2 2.6-1.4 2.4-2.7 3-4.8L1174.6 660l1.1-3.2-.3-2.7-1.5-2.1-2.1-1.5-2.8-1-3.1-.8-2.9-.2-297.3-.1-4.9.1-3.6.6-2.3 1.1-1.9 1.9-1.8 3.3-27 55-6.1 11.5-7.1 11.3-7.6 10.9-8.5 10.7-9.1 10-9.8 9.8-10.1 9.3-10.7 8.8-11.3 8.1-11.5 7.7-12 7.2-12.3 6.4-12.6 5.9-12.6 5.1-13 4.5-13.1 3.8-13.2 2.8-13.1 2.3-13.3 1.2-13 .4zM271.3 425.6H269l-2.2.4-2.3.6-2.4.9-2.3 1-2.3 1.2-2.4 1.4-2.1 1.5-2.3 1.8-2 1.9-1.9 1.9-1.8 2.1-1.5 2.1-1.6 2.1-1.2 2.3-1.2 2.3-.7 2.5-.7 2.3-.2 2.4-.1 2.3.8 3.8 2.8.9 4.6 1.1 4.6.7 527.9 1.5 5.5.2 5.3.4 5.6 1 5.7 1.2 5.5 1.7 5.3 1.8 5.5 2.2 5.4 2.4 5.1 2.8 5 3.2 4.8 3.3 4.6 3.5 4.4 3.8 4.2 4.1 3.7 4.3 3.4 4.4 3.1 4.8 2.8 5 2.3 5.1 1.7 5.1 10 33.6.9 3.3 1.6 1.8 2.2.8 3.8.3 330.8 1.5 3.6-.1 2.8-.4 2.3-1 1.9-1.7 1.9-2.5 2.2-3.6 124.7-219.7 1.3-3.6-.4-3.2-2.1-1.9-4.6-1.7-7.6-1.4-7.6-.5-294.3.4-3.4.2-2.5 1.4-3.1 2.9-41.8 44.5-7.6 6.2-8.4 5.6-4.1 3.3-4.3 3.1-4.5 2.7-4.7 2.7-4.9 2.5-5 2.4-5 2.3-5.1 2.3-5.3 2-5.5 2.2-5.3 1.7-5.4 1.8-5.5 1.6-5.7 1.4-5.5 1.4-5.7 1.2-5.4 1-5.5.8-5.5.8-5.5.7-5.5.4-5.3.3H860zM527.1.9l-3.6.1-5.5.2-6.7.3-7 .7-6.2 1-4.9 1.4-2.5 1.7-14.5 24.1-2.6 5-.5 3.2 1.2 3 1.5 2.3 2.3 1.1 3.8.6h6.3l452.1.8 3.4-.1 3.6.2 3.4.1 3.6.2 3.6.1 3.6.4 3.5.3 3.8.5 3.6.4 3.5.5 3.5.7 3.6.7 3.6.7 3.4 1 3.6.9 3.5.9 3.4 1 3.2 1.3 3.3 1.4 3.2 1.2 7.3 3.6 6.9 4.2 6.7 4.9 6.4 5.2 6 5.8 5.8 6.2 5.2 6.6 5 7.2 4.4 7.3 4.1 7.6 3.7 7.7 3.1 8.1 2.6 8.1 2.2 8.4 1.6 8.3 1.1 8.4.5 8.5-.3 8.2-.7 8.2-1.4 8.1-8.6 39.8-.6 3.4.8 2.1 2.5.9 4.8.2 318.6 2.9h3.8l2.8-.7 2.3-1.3 2-2.3 2.3-3.6 127-220.1 1.5-2.9.7-3v-2.9l-.9-2.6-1.9-1.9-3-1.1-14.1-1-14.4-.4-14-.2z"/></g>
</svg>
"""

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .main-header svg {
        width: 60px;
        height: auto;
    }
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.5rem;
    }
    .sidebar-header svg {
        width: 40px;
        height: auto;
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
        st.session_state.groq_api_key = config.groq_api_key
    if 'nvidia_api_key' not in st.session_state:
        st.session_state.nvidia_api_key = config.nvidia_api_key
    if 'selected_model_ollama' not in st.session_state:
        st.session_state.selected_model_ollama = config.get('ollama.model', 'llama3.2:3b')
    if 'selected_model_gguf' not in st.session_state:
        st.session_state.selected_model_gguf = config.get('local_llm.last_model', None)
    if 'selected_model_groq' not in st.session_state:
        st.session_state.selected_model_groq = config.get('online_api.groq.default_model', 'llama-3.3-70b-versatile')
    if 'selected_model_nvidia' not in st.session_state:
        st.session_state.selected_model_nvidia = config.get('online_api.nvidia.default_model', 'nvidia/llama-3.1-nemotron-70b-instruct')

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


@st.cache_data(ttl=3600, show_spinner="Fetching Nvidia models...")
def get_nvidia_models(api_key):
    """Fetch Nvidia models and cache for 1 hour."""
    from app.core.llm.llm_client import NvidiaClient
    try:
        temp_client = NvidiaClient(api_key=api_key, model="nvidia/llama-3.1-nemotron-70b-instruct")
        return temp_client.list_models()
    except Exception as e:
        logger.error(f"Error fetching Nvidia models: {str(e)}")
        return [
            "nvidia/llama-3.1-nemotron-70b-instruct",
            "meta/llama-3.1-405b-instruct",
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-8b-instruct"
        ]

def render_sidebar():
    with st.sidebar:
        st.markdown(f"<div class='sidebar-header'>{LOGO_SVG} <span style='font-size: 2rem; font-weight: bold; color: #1f77b4;'>ILIN</span></div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Integrated Localized Intelligence Node</div>", unsafe_allow_html=True)
        
        # 1. Engine Selection
        st.markdown("### 🔌 LLM Engine")
        
        # Define available engines - easy to add new ones
        ENGINE_OPTIONS = ["Ollama", "Local GGUF", "Groq (Online)", "Nvidia (Online)"]
        
        # Validate and reset session state if current value is invalid
        # This prevents errors when option text changes or new options are added
        current_engine = st.session_state.get("engine_type")
        if current_engine not in ENGINE_OPTIONS:
            # Try to get last saved engine from config, or default to first option
            last_engine = config.get("last_engine")
            if last_engine in ENGINE_OPTIONS:
                st.session_state.engine_type = last_engine
            else:
                st.session_state.engine_type = ENGINE_OPTIONS[0]
        
        engine_type = st.selectbox(
            "Select Engine",
            ENGINE_OPTIONS,
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

        elif engine_type == "Groq (Online)":
            st.markdown("#### Groq Settings")
            api_key = st.text_input("Groq API Key", type="password", key="groq_api_key")
            
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

        elif engine_type == "Nvidia (Online)":
            st.markdown("#### Nvidia Settings")
            api_key = st.text_input("Nvidia API Key", type="password", key="nvidia_api_key")

            if api_key:
                available_models = get_nvidia_models(api_key)

                # Check if current st.session_state value is valid
                if st.session_state.selected_model_nvidia not in available_models:
                    st.session_state.selected_model_nvidia = available_models[0]

                selected_model = st.selectbox(
                    "Select Model",
                    available_models,
                    key="selected_model_nvidia"
                )

                # Persistence
                if selected_model != config.get('online_api.nvidia.default_model'):
                    config.set('online_api.nvidia.default_model', selected_model)
                    config.save()
            else:
                st.warning("⚠️ Please enter your Nvidia API Key.")

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
            elif engine_type == "Groq (Online)" and selected_model and api_key:
                set_llm_client("Groq (Online)", api_key=api_key, model=selected_model)
                st.session_state.engine_ready = True
            elif engine_type == "Nvidia (Online)" and selected_model and api_key:
                set_llm_client("Nvidia (Online)", api_key=api_key, model=selected_model)
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
        
        # --- Knowledge Management in Sidebar ---
        st.divider()
        st.markdown("### 📁 Knowledge Management")
        
        with st.expander("📤 Upload Documents"):
            uploaded_files = st.file_uploader(
                "Drag and drop files",
                type=['txt', 'pdf', 'docx', 'md', 'html', 'csv'],
                accept_multiple_files=True,
                key="sidebar_uploader"
            )
            
            if uploaded_files:
                for file in uploaded_files:
                    st.caption(f"📄 {file.name} ({(file.size/1024/1024):.1f}MB)")
                
                if st.button("🚀 Process Files", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    for idx, file in enumerate(uploaded_files):
                        try:
                            documents_path = Path(config.documents_path)
                            documents_path.mkdir(parents=True, exist_ok=True)
                            
                            file_path = documents_path / file.name
                            with open(file_path, "wb") as f:
                                f.write(file.getvalue())
                            
                            rag_pipeline.ingest_document(str(file_path))
                            progress_bar.progress((idx + 1) / len(uploaded_files))
                        except Exception as e:
                            st.error(f"Error: {file.name} - {str(e)}")
                    
                    st.cache_data.clear()
                    st.success("Indexing complete!")
                    time.sleep(1)
                    st.rerun()

        with st.expander("📋 Document Library"):
            try:
                documents = rag_pipeline.list_documents()
                if not documents:
                    st.info("No documents found.")
                else:
                    for doc in documents:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{doc['filename']}**")
                            st.caption(f"{doc['total_chunks']} chunks | {(doc['file_size']/1024/1024):.2f} MB")
                        with col2:
                            if st.button("🗑️", key=f"side_del_{doc['document_id']}"):
                                if rag_pipeline.delete_document(doc['document_id']):
                                    st.cache_data.clear()
                                    st.rerun()
                        st.divider()
            except Exception as e:
                st.error(f"Error loading: {str(e)}")
        # --------------------------------------

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
    st.markdown(f"<div class='main-header'>{LOGO_SVG} Chat with Your Equipment</div>", unsafe_allow_html=True)
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
    elif st.session_state.engine_type == "Groq (Online)":
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
        # Add user message to history and display it immediately
        st.session_state.chat_history.append({
            'role': 'user',
            'content': prompt
        })
        
        # Rerun to show user message first
        st.rerun()
    
    # Check if we need to generate a response (last message is from user)
    if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'user':
        prompt = st.session_state.chat_history[-1]['content']
        
        # Get chat history (all messages except the last user message)
        chat_history = st.session_state.chat_history[:-1] if len(st.session_state.chat_history) > 1 else []
        
        # Display the analyzing message
        with st.chat_message("assistant"):
            analysis_msg = st.empty()
            analysis_msg.markdown("*Analysing the knowledge base...*")
            
            try:
                response = rag_pipeline.query(
                    question=prompt,
                    search_mode=search_mode,
                    top_k=top_k,
                    stream=False,
                    use_rag=use_rag,
                    chat_history=chat_history
                )
                
                # Clear the analyzing message
                analysis_msg.empty()
                
                answer = response['answer']
                sources = response['sources']
                
                # Generate the answer character by character for live effect
                displayed_answer = ""
                answer_placeholder = st.empty()
                
                for char in answer:
                    displayed_answer += char
                    answer_placeholder.markdown(displayed_answer)
                    time.sleep(0.01)  # Small delay for typing effect
                
                # Display sources if available
                if sources:
                    with st.expander("📚 View Sources"):
                        for source in sources:
                            st.markdown(f"""
                            <div class='source-box'>
                                <strong>{source['filename']}</strong> 
                                <span class='score-badge'>Score: {source['relevance_score']:.3f}</span>
                                <br><small>Type: {source['search_type']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': answer,
                    'sources': sources
                })
                
            except Exception as e:
                analysis_msg.empty()
                st.error(f"Error: {str(e)}")
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': f"Error: {str(e)}",
                    'sources': []
                })
        
        # Rerun to display the final formatted message
        st.rerun()
    
    if st.button("🗑️ Clear Conversation", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()


def render_search_page():
    st.markdown(f"<div class='main-header'>{LOGO_SVG} Advanced Search</div>", unsafe_allow_html=True)
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
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💬 Chat", use_container_width=True, type="primary" if st.session_state.active_tab == "Chat" else "secondary"):
            if st.session_state.active_tab != "Chat":
                st.session_state.active_tab = "Chat"
                st.rerun()
    with col2:
        if st.button("🔍 Search", use_container_width=True, type="primary" if st.session_state.active_tab == "Search" else "secondary"):
            if st.session_state.active_tab != "Search":
                st.session_state.active_tab = "Search"
                st.rerun()
    
    st.divider()
    
    # Render the active tab
    if st.session_state.active_tab == "Chat":
        render_chat_page(search_mode, top_k, use_rag)
    elif st.session_state.active_tab == "Search":
        render_search_page()

if __name__ == "__main__":
    main()
