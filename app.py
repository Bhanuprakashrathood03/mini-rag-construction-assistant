import streamlit as st
from src.query_engine import QueryEngine
import os

# Paths
INDEX_PATH = "faiss_index/indecimal_index.faiss"
META_PATH = "faiss_index/metadata.pkl"

# Streamlit setup
st.set_page_config(page_title="🏗️ Mini RAG - Indecimal", page_icon="🏗️")
st.title("🏗️ Indecimal Construction Assistant (Ollama-Powered)")
st.caption("💡 Ask questions about Indecimal’s services, quality, or AI-driven construction workflow.")

# Check for FAISS index and metadata
if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    st.error("❌ FAISS index or metadata not found. Please run `python src/build_index.py` first.")
    st.stop()

# Cache Query Engine
@st.cache_resource
def load_engine():
    return QueryEngine(INDEX_PATH, META_PATH)

qe = load_engine()

# User input
query = st.text_input("🔍 Ask a question:")

# Query handler
if st.button("Ask") and query.strip():
    with st.spinner("💬 Generating answer using Llama 3 (Ollama)..."):
        st.markdown("### 🧠 Answer:")
        placeholder = st.empty()
        partial_text = ""

        # Stream the model output chunk by chunk
        for chunk in qe.stream_ollama_response(query):
            # Clean up escaped unicode & markdown characters
            chunk = (
                chunk.replace("\\n", "\n")
                     .replace("\\u0026", "&")
                     .replace("\\u2019", "'")
                     .replace("\\u201c", '"')
                     .replace("\\u201d", '"')
            )

            partial_text += chunk
            placeholder.markdown(partial_text, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("🤖 Powered by Llama 3 via Ollama | FAISS + Sentence Transformers")
