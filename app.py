import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import List, Sequence

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NeMo Guardrails imports
try:
    from nemoguardrails import RailsConfig
    from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False

# --- CONFIGURATION & PATHS ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
FAISS_DIR = INDEX_DIR / "faiss"
BM25_PATH = INDEX_DIR / "bm25.pkl"
META_PATH = INDEX_DIR / "metadata.json"
NEMO_CONFIG_DIR = BASE_DIR / "nemo_config"

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

def get_all_local_pdfs() -> List[Path]:
    """Scan both data/pdf and data/pdfs automatically for documents."""
    candidate_folders = [DATA_DIR / "pdf", DATA_DIR / "pdfs", DATA_DIR]
    discovered = set()
    for folder in candidate_folders:
        if folder.exists():
            for p in folder.rglob("*"):
                if p.is_file() and p.suffix.lower() == ".pdf" and "index" not in p.parts:
                    discovered.add(p.resolve())
    return sorted(list(discovered), key=lambda p: p.name.lower())

load_dotenv()

# --- PAGE STYLING & BRANDING ---
st.set_page_config(
    page_title="TN Agri Schemes Intelligence Portal",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for larger fonts, executive-grade styling, and citation cards
st.markdown("""
<style>
    /* Increase global font size */
    html, body, [class*="css"] {
        font-size: 17px !important;
    }
    
    /* Metric pill styling */
    .metric-badge {
        display: inline-block;
        padding: 6px 12px;
        background-color: #e8f5e9;
        color: #2e7d32;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    
    /* Citation Cards Formatting */
    .citation-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .citation-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-weight: 600;
        font-size: 0.9rem;
        color: #1e293b;
        margin-bottom: 6px;
    }
    .citation-badge {
        background-color: #dcfce7;
        color: #15803d;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .citation-snippet {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.5;
        font-style: italic;
    }
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

DEFAULT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

SYSTEM_PROMPT = """You are an official intelligence assistant for the Tamil Nadu Department of Agriculture and Farmers Welfare.
Your task is to provide verified answers about Tamil Nadu agricultural schemes, subsidies, eligibility criteria, and farmer welfare programs (e.g., Kalaignarin All Village Integrated Agriculture Development Programme, Micro Irrigation Subsidy, CM Solar Pump Sets, PMFBY, Soil Health Cards).

Answer using ONLY the supplied policy document excerpts. If the excerpts do not contain enough information, state clearly that the specific detail was not found in the indexed policy records.
Never fabricate subsidy amounts, financial percentages, criteria, or citations.
Cite supporting evidence inline using.
Keep answers structured, clear, and actionable.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nPolicy Document Excerpts:\n{context}\n\nAnswer with inline citations.")
])

# --- HELPER FUNCTIONS ---
def get_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()

def file_fingerprint_from_paths(pdf_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for pdf_path in sorted(pdf_paths, key=lambda p: p.name.lower()):
        digest.update(pdf_path.name.encode("utf-8"))
        digest.update(pdf_path.read_bytes())
    return digest.hexdigest()

def normalize_page_number(page_doc: Document) -> int:
    try:
        return int(page_doc.metadata.get("page", 0)) + 1
    except (TypeError, ValueError):
        return 1

def load_pdf_documents(pdf_paths: Sequence[Path]) -> List[Document]:
    documents = []
    for pdf_path in pdf_paths:
        try:
            pages = PyPDFLoader(str(pdf_path)).load()
            for page_doc in pages:
                page_doc.metadata["source"] = pdf_path.name
                page_doc.metadata["page_number"] = normalize_page_number(page_doc)
                documents.append(page_doc)
        except Exception:
            continue
    return documents

def ensure_nemo_config():
    if not NEMO_CONFIG_DIR.exists():
        NEMO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        # Using encoding="utf-8" to fix Windows Tamil encoding errors
        with open(NEMO_CONFIG_DIR / "config.yml", "w", encoding="utf-8") as f:
            f.write(f"models:\n  - type: main\n    engine: openai\n    model: {DEFAULT_MODEL}\n")
        with open(NEMO_CONFIG_DIR / "rails.co", "w", encoding="utf-8") as f:
            f.write("""define user ask off topic
  "What is the weather?"
  "Write python code"
  "Tell me a joke"
  "Who won the match?"

define bot refuse off topic
  "I am specialized strictly in Tamil Nadu Agricultural Schemes (தமிழ்நாடு வேளாண்மை திட்டங்கள்). I cannot address unrelated queries."

define flow off topic
  user ask off topic
  bot refuse off topic
""")

def unpack_stream(generator):
    """Safely extracts text chunks from strings, dicts, or AIMessage objects for UI rendering."""
    for chunk in generator:
        if isinstance(chunk, dict):
            content = chunk.get("output", "")
        elif hasattr(chunk, "content"):
            content = str(chunk.content)
        else:
            content = str(chunk)
        if content:
            yield content

# --- INDEX MANAGEMENT ---
def load_saved_index():
    if not (FAISS_DIR / "index.faiss").exists() or not BM25_PATH.exists() or not META_PATH.exists():
        return None, None, None
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    faiss_store = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
    
    with open(BM25_PATH, "rb") as f:
        bm25_retriever = pickle.load(f)
        
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
        
    return faiss_store, bm25_retriever, metadata

def build_vector_store_from_documents(documents: List[Document], chunk_size: int, chunk_overlap: int, fingerprint: str):
    if not documents:
        raise ValueError("No readable PDF records available to index.")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    faiss_store = FAISS.from_documents(chunks, embeddings)
    faiss_store.save_local(str(FAISS_DIR))
    
    bm25_retriever = BM25Retriever.from_documents(chunks)
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    metadata = {
        "fingerprint": fingerprint,
        "page_count": len(documents),
        "chunk_count": len(chunks),
        "files": sorted({d.metadata.get("source", "Unknown") for d in documents})
    }
    with open(META_PATH, "w") as f:
        json.dump(metadata, f)
        
    return faiss_store, bm25_retriever, metadata

# --- STATE INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "faiss_store" not in st.session_state:
    st.session_state.faiss_store = None
    st.session_state.bm25_retriever = None
    st.session_state.index_metadata = None

folder_pdf_files = get_all_local_pdfs()
current_fingerprint = file_fingerprint_from_paths(folder_pdf_files) if folder_pdf_files else None

# Auto-load or Auto-index on Startup
if st.session_state.faiss_store is None:
    api_key = get_api_key()
    if not api_key:
        st.sidebar.error("❌ OPENAI_API_KEY not found in .env! Add it so indexing can proceed.")
    else:
        try:
            faiss_store, bm25, meta = load_saved_index()
            if faiss_store is not None:
                st.session_state.faiss_store = faiss_store
                st.session_state.bm25_retriever = bm25
                st.session_state.index_metadata = meta
            elif folder_pdf_files:
                with st.spinner(f"Found {len(folder_pdf_files)} PDF(s). Indexing knowledge base now..."):
                    docs = load_pdf_documents(folder_pdf_files)
                    f_store, b_ret, meta = build_vector_store_from_documents(docs, 1000, 150, current_fingerprint)
                    st.session_state.faiss_store = f_store
                    st.session_state.bm25_retriever = b_ret
                    st.session_state.index_metadata = meta
                    st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Initialization error: {exc}")

# --- SIDEBAR: SYSTEM CONTROLS & REPOSITORY ---
with st.sidebar:
    st.title("🌾 Department of Agriculture & Farmers Welfare")
    st.caption("Govt. of Tamil Nadu")
    st.divider()

    # Index Status Card
    st.subheader("Repository Status")
    if st.session_state.index_metadata:
        meta = st.session_state.index_metadata
        st.markdown(f"""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
            <div style="font-size: 0.85rem; color: #64748b;">Active Index</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a;">{meta.get('page_count', 0)} Pages Verified</div>
            <div style="font-size: 0.8rem; color: #475569; margin-top: 4px;"><b>{meta.get('chunk_count', 0)}</b> Hybrid Chunks across <b>{len(meta.get('files', []))}</b> Docs</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("No knowledge base loaded. Use the options below to index documents.")

    # Found Documents Log
    if folder_pdf_files:
        with st.expander(f"📂 Discovered Local PDFs ({len(folder_pdf_files)})", expanded=False):
            for f in folder_pdf_files:
                st.caption(f"• {f.name}")

    # Upload Additional Documents
    st.subheader("Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload additional Policy PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload Government Orders (GOs), scheme guidelines, or circulars."
    )

    # Advanced Tuning Expander
    with st.expander("Retrieval & Guardrail Settings"):
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 400, 150, 25)
        top_k = st.slider("Final Verified Chunks", 2, 10, 4)
        use_guardrails = st.checkbox("NeMo Topical Guardrails", value=NEMO_AVAILABLE, disabled=not NEMO_AVAILABLE)

    # Rebuild Index Button
    can_rebuild = bool(folder_pdf_files or uploaded_files)
    if st.button("Rebuild Knowledge Base", type="primary", disabled=not can_rebuild, use_container_width=True):
        if not get_api_key():
            st.error("OPENAI_API_KEY is not set.")
        else:
            with st.spinner("Processing documents, generating embeddings, and building hybrid index..."):
                all_docs = load_pdf_documents(folder_pdf_files)
                if uploaded_files:
                    for up_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(up_file.getvalue())
                            tmp_path = tmp.name
                        try:
                            pages = PyPDFLoader(tmp_path).load()
                            for p in pages:
                                p.metadata["source"] = up_file.name
                                p.metadata["page_number"] = normalize_page_number(p)
                                all_docs.append(p)
                        finally:
                            os.unlink(tmp_path)
                            
                f_store, b_ret, meta = build_vector_store_from_documents(all_docs, chunk_size, chunk_overlap, current_fingerprint)
                st.session_state.faiss_store = f_store
                st.session_state.bm25_retriever = b_ret
                st.session_state.index_metadata = meta
                st.session_state.chat_history = []
                st.success("Repository index successfully rebuilt.")
                st.rerun()

    if st.session_state.chat_history:
        st.divider()
        if st.button("Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# --- MAIN PAGE: HEADER & CHIPS ---
st.title("🌾 Tamil Nadu Agri Schemes Assistant")
st.markdown("""
**தமிழ்நாடு வேளாண்மை - உழவர் நலத்துறை தகவல் மற்றும் வழிகாட்டு மையம்**  
Instant, citation-backed answers on state agricultural subsidies, GO criteria, and farmer assistance schemes.
""")

# Guided Query Chips
st.markdown("""
<div style="margin-top: 10px; margin-bottom: 20px;">
    <span class="metric-badge">💡 Micro Irrigation 100% Subsidies</span>
    <span class="metric-badge">💡 CM Solar Powered Pump Sets</span>
    <span class="metric-badge">💡 PMFBY Crop Insurance Deadlines</span>
    <span class="metric-badge">💡 Kuruvai Cultivation Packages</span>
</div>
""", unsafe_allow_html=True)
st.divider()

# --- CHAT DISPLAY ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        content = message["content"]
        # Handle dict extraction for historical messages
        if isinstance(content, dict):
            content = content.get("output", str(content))
        st.markdown(content)

        if message.get("sources"):
            with st.expander("📑 View Reference Citations", expanded=False):
                for s in message["sources"]:
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">
                            <span>📄 {s['source']}</span>
                            <span class="citation-badge">Page {s['page']}</span>
                        </div>
                        <div class="citation-snippet">"{s['text']}"</div>
                    </div>
                    """, unsafe_allow_html=True)

# --- QUERY EXECUTION & STREAMING ---
user_query = st.chat_input("உங்களுக்கு என்ன திட்டம் அல்லது மானியம் பற்றிய தகவல் தேவை? / Ask about any TN scheme...")

if user_query:
    if not get_api_key():
        st.error("Please configure your OPENAI_API_KEY in the .env file.")
    elif st.session_state.faiss_store is None:
        st.warning("The knowledge base is empty. Please verify policy PDFs exist or upload them via the sidebar.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            # 1. Hybrid Search
            st.session_state.bm25_retriever.k = 15
            faiss_retriever = st.session_state.faiss_store.as_retriever(search_kwargs={"k": 15})
            ensemble = EnsembleRetriever(
                retrievers=[st.session_state.bm25_retriever, faiss_retriever],
                weights=[0.4, 0.6]
            )

            # 2. Cross-Encoder Reranking
            try:
                compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=top_k)
                retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)
            except Exception:
                retriever = ensemble

            with st.spinner("Searching scheme guidelines and validating citations..."):
                retrieved_docs = retriever.invoke(user_query)

            context_text = "\n\n".join([f"[Excerpt {i+1} | Source: {d.metadata.get('source', 'Doc')}, p. {d.metadata.get('page_number', 1)}]\n{d.page_content}" for i, d in enumerate(retrieved_docs)])

            # 3. Model Chain & Guardrails
            llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0, streaming=True)
            core_chain = PROMPT | llm | StrOutputParser()

            if use_guardrails and NEMO_AVAILABLE:
                ensure_nemo_config()
                rails_config = RailsConfig.from_path(str(NEMO_CONFIG_DIR))
                guardrails = RunnableRails(config=rails_config)
                active_chain = guardrails | core_chain
            else:
                active_chain = core_chain

            # 4. Streamed Output (Unpacking Dict to Markdown)
            try:
                raw_stream = active_chain.stream({"question": user_query, "context": context_text})
                answer = st.write_stream(unpack_stream(raw_stream))
                
                # Double guard: in case write_stream returned a dict or json-like object
                if isinstance(answer, dict):
                    answer = answer.get("output", str(answer))
            except Exception as e:
                answer = f"Error processing query: {e}"
                st.error(answer)

            # 5. Extract Citations & Render in Styled Cards
            citations = []
            if retrieved_docs:
                st.markdown("#### 📑 Verified Policy Citations")
                for idx, d in enumerate(retrieved_docs, start=1):
                    src = d.metadata.get("source", "Policy Document")
                    pg = d.metadata.get("page_number", "N/A")
                    clean_excerpt = d.page_content.replace("\n", " ").strip()
                    if len(clean_excerpt) > 320:
                        clean_excerpt = clean_excerpt[:320] + "..."
                    
                    citations.append({"source": src, "page": pg, "text": clean_excerpt})

                    # Render card component
                    st.markdown(f"""
                    <div class="citation-card">
                        <div class="citation-header">
                            <span>📄 {src}</span>
                            <span class="citation-badge">Page {pg}</span>
                        </div>
                        <div class="citation-snippet">"{clean_excerpt}"</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Save clean text to session state
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": str(answer),
                "sources": citations
            })