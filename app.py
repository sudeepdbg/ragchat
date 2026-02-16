import streamlit as st
import os
import tempfile
from pathlib import Path
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import requests

# ---------- Configuration ----------
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL = "llama3.2"                 # or "phi3:mini", "mistral", etc.
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "my_docs_collection"
CHUNK_SIZE = 500                        # words per chunk
CHUNK_OVERLAP = 50                       # overlap between chunks
TOP_K_RETRIEVAL = 5                      # number of chunks to retrieve

# ---------- Helper: split text into overlapping chunks ----------
def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of approximately `chunk_size` words."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i+chunk_size])
        # Only keep chunks with meaningful content (adjust threshold as needed)
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks

# ---------- Check Ollama availability ----------
def is_ollama_available():
    """Check if Ollama server is running."""
    try:
        ollama.list()
        return True
    except (requests.exceptions.ConnectionError, Exception):
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

        chunks = split_text(full_text)  # use the improved chunker
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
            # Fallback to showing retrieved chunks if generation fails
            st.warning("⚠️ Ollama encountered an error. Showing retrieved context instead.")
            answer = "**Relevant excerpts from your documents:**\n\n"
            for i, doc in enumerate(retrieved_docs[:3]):  # show top 3
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
st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
st.title("📚 Chat with Your Documents (Local RAG)")

# Check Ollama availability once and store in session state
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
