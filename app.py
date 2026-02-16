import streamlit as st
import os
import tempfile
import shutil
import re
import requests
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ---------- Configuration ----------
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL = "llama3.2"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "my_docs_collection"
CHUNK_SIZE = 250          # words per chunk – smaller for precision
CHUNK_OVERLAP = 30        # overlap between chunks
TOP_K_RETRIEVAL = 5

# ---------- Intelligent chunking (paragraph + sentence aware) ----------
def split_into_paragraphs(text):
    """Split text by double newlines or common paragraph breaks."""
    # Normalize line endings
    text = re.sub(r'\r\n', '\n', text)
    # Split on one or more blank lines (paragraph separator)
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]

def split_paragraph_into_sentences(para):
    """Simple sentence splitter (handles .!? followed by space)."""
    sentences = re.split(r'(?<=[.!?])\s+', para)
    return [s.strip() for s in sentences if s.strip()]

def smart_chunking(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Recursively split text into chunks of approximately `chunk_size` words,
    preserving paragraph and sentence boundaries as much as possible.
    """
    # First split into paragraphs
    paragraphs = split_into_paragraphs(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        # Split paragraph into sentences
        sentences = split_paragraph_into_sentences(para)
        for sent in sentences:
            word_count = len(sent.split())
            # If adding this sentence exceeds chunk size and we already have content, finalize chunk
            if current_word_count + word_count > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                # Keep last `overlap` words from previous chunk for overlap
                overlap_words = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk = overlap_words + [sent]
                current_word_count = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sent)
                current_word_count += word_count

    # Add last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# ---------- Check Ollama availability ----------
def is_ollama_available():
    """Robust check using both library and direct HTTP."""
    try:
        ollama.list()
        return True
    except:
        pass
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except:
        return False

# ---------- Initialize Embedder ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

embedder = load_embedder()

# ---------- Initialize ChromaDB ----------
@st.cache_resource
def init_chromadb():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except:
        collection = client.create_collection(name=COLLECTION_NAME)
    return client, collection

client, collection = init_chromadb()

# ---------- Process uploaded PDF ----------
def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        reader = pypdf.PdfReader(tmp_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        chunks = smart_chunking(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            st.warning(f"No text chunks extracted from {uploaded_file.name}")
            return 0

        ids = []
        texts = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            ids.append(f"{uploaded_file.name}_{i}")
            texts.append(chunk)
            metadatas.append({"source": uploaded_file.name})

        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return len(chunks)
    finally:
        os.unlink(tmp_path)

# ---------- Clear database ----------
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    st.cache_resource.clear()
    st.session_state.messages = []
    st.rerun()

# ---------- Ask a question ----------
def ask_question(question, top_k=TOP_K_RETRIEVAL):
    q_emb = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    if not results['documents'][0]:
        return "I couldn't find any relevant information.", []

    retrieved_docs = results['documents'][0]
    sources = list(set([m['source'] for m in results['metadatas'][0]]))

    # If Ollama is available, generate an answer using all retrieved chunks
    if st.session_state.ollama_available:
        context = "\n\n---\n\n".join(retrieved_docs)
        prompt = f"""You are a helpful assistant. Answer the question based *only* on the context below.
If the answer is not found in the context, say "I cannot find the answer in the provided documents."

Context:
{context}

Question: {question}
Answer:"""

        try:
            response = ollama.generate(model=LLM_MODEL, prompt=prompt)
            return response['response'], sources
        except Exception as e:
            st.warning(f"⚠️ Ollama error: {e}. Showing retrieved context instead.")
            answer = "**Relevant excerpts from your documents:**\n\n"
            for i, doc in enumerate(retrieved_docs[:3]):
                answer += f"--- Excerpt {i+1} ---\n{doc}\n\n"
            return answer, sources
    else:
        # Ollama not available – show top retrieved chunks
        answer = "**Relevant excerpts from your documents (Ollama offline):**\n\n"
        for i, doc in enumerate(retrieved_docs[:3]):
            answer += f"--- Excerpt {i+1} ---\n{doc}\n\n"
        if len(retrieved_docs) > 3:
            answer += f"\n*(showing 3 of {len(retrieved_docs)} relevant excerpts)*"
        return answer, sources

# ---------- Streamlit UI ----------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 Chat with Your Documents (RAG)")

# Check Ollama availability
if "ollama_available" not in st.session_state:
    st.session_state.ollama_available = is_ollama_available()

with st.sidebar:
    st.header("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        for file in uploaded_files:
            existing = collection.get(where={"source": file.name})
            if existing['ids']:
                st.info(f"✅ {file.name} already in database.")
            else:
                with st.spinner(f"Processing {file.name}..."):
                    num_chunks = process_uploaded_file(file)
                    if num_chunks:
                        st.success(f"Added {num_chunks} chunks from {file.name}")
                    else:
                        st.error(f"Failed to process {file.name}")

    st.divider()
    if st.button("🗑️ Clear Database"):
        clear_database()

    st.divider()
    status = "✅ Online (Ollama ready)" if st.session_state.ollama_available else "❌ Offline (showing raw excerpts)"
    st.caption(f"**Ollama status:** {status}")
    st.caption(f"Using model: **{LLM_MODEL}**")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            st.caption(f"📁 Sources: {', '.join(msg['sources'])}")

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer, sources = ask_question(prompt)
            st.markdown(answer)
            if sources:
                st.caption(f"📁 Sources: {', '.join(sources)}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
