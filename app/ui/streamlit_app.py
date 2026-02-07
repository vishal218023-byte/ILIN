import streamlit as st
import requests
import json
from pathlib import Path
import time

from app.core.config import config
from app.core.rag_pipeline import rag_pipeline
from app.core.ollama_client import ollama_client

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
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #1f77b4;
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
        background-color: white;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'ollama_status' not in st.session_state:
        st.session_state.ollama_status = False

def check_ollama_status():
    try:
        return ollama_client.check_model_available()
    except:
        return False

def render_sidebar():
    with st.sidebar:
        st.markdown("<div class='main-header'>🧠 ILIN</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Integrated Localized Intelligence Node</div>", unsafe_allow_html=True)
        
        st.session_state.ollama_status = check_ollama_status()
        
        if st.session_state.ollama_status:
            st.success("✅ Ollama Connected")
        else:
            st.error("❌ Ollama Not Available")
            st.info("Make sure Ollama is running locally")
        
        st.divider()
        
        try:
            stats = rag_pipeline.get_stats()
            st.markdown("### 📊 Knowledge Base Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats['total_documents'])
            with col2:
                st.metric("Chunks", stats['total_chunks'])
            st.caption(f"Index Size: {stats['index_size']} vectors")
        except:
            st.info("No documents indexed yet")
        
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
            st.success("Index refreshed!")
        
        return search_mode, top_k

def render_chat_page(search_mode, top_k):
    st.markdown("<div class='main-header'>💬 Chat with Your Documents</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Ask questions and get answers based on your knowledge base</div>", unsafe_allow_html=True)
    
    if not st.session_state.ollama_status:
        st.warning("⚠️ Ollama is not connected. Please start Ollama first.")
        return
    
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
    
    if prompt := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.chat_history.append({
            'role': 'user',
            'content': prompt
        })
        
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating answer..."):
                try:
                    response = rag_pipeline.query(
                        question=prompt,
                        search_mode=search_mode,
                        top_k=top_k,
                        stream=False
                    )
                    
                    answer = response['answer']
                    sources = response['sources']
                    
                    st.markdown(answer)
                    
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
                    
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': answer,
                        'sources': sources
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
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
    
    search_mode, top_k = render_sidebar()
    
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📁 Documents", "🔍 Search"])
    
    with tab1:
        render_chat_page(search_mode, top_k)
    
    with tab2:
        render_documents_page()
    
    with tab3:
        render_search_page()

if __name__ == "__main__":
    main()
