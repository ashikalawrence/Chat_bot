import streamlit as st
import requests
import os
import uuid

from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb


# =====================================
# 1. CONFIGURATION
# =====================================

load_dotenv()

OLLAMA_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3.2"
)

EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text"
)

CHROMA_PATH = os.getenv(
    "CHROMA_PATH",
    "chroma_db"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))
TOP_K = int(os.getenv("TOP_K", 5))


# =====================================
# 2. STREAMLIT PAGE
# =====================================

st.set_page_config(
    page_title="Knowledge AI",
    page_icon="🧠",
    layout="wide"
)


# =====================================
# 3. CUSTOM UI
# =====================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0b1020,
        #15152f,
        #101b35
    );
    color: white;
}

[data-testid="stSidebar"] {
    background: #11162b;
}

.main-title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    background: linear-gradient(
        90deg,
        #a78bfa,
        #60a5fa,
        #22d3ee
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #a5b4fc;
    margin-bottom: 30px;
}

.welcome-card {
    background: linear-gradient(
        135deg,
        #27204c,
        #172c50
    );
    border: 1px solid #55558a;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    margin-bottom: 25px;
}

.welcome-card h2 {
    color: #ddd6fe;
}

.welcome-card p {
    color: #cbd5e1;
    line-height: 1.8;
}

[data-testid="stChatMessage"] {
    background: rgba(40, 45, 80, 0.8);
    border: 1px solid #42466d;
    border-radius: 15px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# 4. CHROMADB CONNECTION
# =====================================

@st.cache_resource
def get_database():

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name="pdf_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    return collection


collection = get_database()


# =====================================
# 5. CREATE EMBEDDINGS
# =====================================

def create_embeddings(texts):

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()["embeddings"]


# =====================================
# 6. PDF TEXT EXTRACTION
# =====================================

def extract_pdf(file):

    reader = PdfReader(file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            pages.append({
                "text": text,
                "page": page_number
            })

    return pages


# =====================================
# 7. TEXT CHUNKING
# =====================================

def split_text(text):

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# =====================================
# 8. PROCESS PDF
# =====================================

def process_pdf(file):

    pages = extract_pdf(file)

    all_chunks = []

    for page in pages:

        chunks = split_text(page["text"])

        for chunk in chunks:

            all_chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    if not all_chunks:

        raise ValueError(
            "No readable text found in this PDF."
        )

    batch_size = 32

    for start in range(
        0,
        len(all_chunks),
        batch_size
    ):

        batch = all_chunks[
            start:start + batch_size
        ]

        texts = [
            item["text"] for item in batch
        ]

        embeddings = create_embeddings(texts)

        ids = [
            str(uuid.uuid4()) for _ in batch
        ]

        metadatas = [
            {
                "source": file.name,
                "page": item["page"]
            }
            for item in batch
        ]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    return len(all_chunks)


# =====================================
# 9. RETRIEVAL
# =====================================

def retrieve_documents(question):

    query_embedding = create_embeddings(
        [question]
    )[0]

    count = collection.count()

    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, count),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        documents.append({
            "text": document,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": distance
        })

    return documents


# =====================================
# 10. LLAMA 3.2 RESPONSE
# =====================================

def generate_answer(question, documents):

    context = ""

    for index, doc in enumerate(
        documents,
        start=1
    ):

        context += f"""
        Source {index}
        File: {doc['source']}
        Page: {doc['page']}
        Content: {doc['text']}

        """

    prompt = f"""
    You are Knowledge AI.

    Answer the user's question using only the
    information provided in the PDF context.

    If the answer is not available in the PDF,
    clearly say that the information was not found.

    PDF CONTEXT:

    {context}

    USER QUESTION:

    {question}

    ANSWER:
    """

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


# =====================================
# 11. SESSION STATE
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================
# 12. SIDEBAR
# =====================================

with st.sidebar:

    st.markdown(
        "<h1 style='color:#a78bfa;'>🧠 Knowledge AI</h1>",
        unsafe_allow_html=True
    )

    st.caption("Your Personal PDF Assistant")

    st.divider()

    st.markdown("### 📂 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF document",
        type=["pdf"]
    )

    if st.button(
        "⚡ Process PDF",
        use_container_width=True
    ):

        if uploaded_file is None:

            st.warning("Please upload a PDF first.")

        else:

            try:

                with st.spinner(
                    "Reading and processing PDF..."
                ):

                    count = process_pdf(uploaded_file)

                st.success(
                    f"Successfully stored {count} chunks!"
                )

            except Exception as error:

                st.error(f"Error: {error}")

    st.divider()

    st.markdown("### 📊 Knowledge Base")

    st.metric(
        "Stored Chunks",
        collection.count()
    )

    if st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True
    ):

        existing = collection.get()

        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        st.session_state.messages = []

        st.success("Knowledge base cleared!")

        st.rerun()

    st.divider()

    st.markdown("### 🤖 AI Models")

    st.info(f"LLM: {LLM_MODEL}")

    st.info(f"Embedding: {EMBED_MODEL}")

    st.success("Vector DB: ChromaDB")


# =====================================
# 13. MAIN HEADER
# =====================================

st.markdown(
    "<div class='main-title'>🧠 KNOWLEDGE AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>"
    "Chat with your PDF documents using Artificial Intelligence"
    "</div>",
    unsafe_allow_html=True
)


# =====================================
# 14. WELCOME SCREEN
# =====================================

if not st.session_state.messages:

    st.html("""
    <div class="welcome-card">

        <h2>👋 Welcome to Knowledge AI!</h2>

        <p>
        Upload your PDF document.

        <br><br>

        I will read your document, understand its content
        and help you find the information you need.

        <br><br>

        Ask questions and get intelligent answers
        based on your PDF.

        </p>

        <h2>✨ Your Documents. Your Knowledge. Your AI.</h2>

    </div>
    """)

    st.markdown("### 💡 Example Questions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("📄 Summarize my PDF")

    with col2:
        st.info("🔍 Find important information")

    with col3:
        st.info("📚 Explain a topic")


# =====================================
# 15. CHAT HISTORY
# =====================================

for message in st.session_state.messages:

    avatar = (
        "🧑‍💻"
        if message["role"] == "user"
        else "🧠"
    )

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.markdown(message["content"])


# =====================================
# 16. USER QUESTION
# =====================================

question = st.chat_input(
    "Ask anything about your PDF..."
)


# =====================================
# 17. RAG PIPELINE
# =====================================

if question:

    if collection.count() == 0:

        st.warning(
            "Please upload and process a PDF first."
        )

    else:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user", avatar="🧑‍💻"):

            st.markdown(question)

        with st.chat_message("assistant", avatar="🧠"):

            try:

                with st.spinner(
                    "Searching PDF and generating answer..."
                ):

                    retrieved_docs = retrieve_documents(
                        question
                    )

                    answer = generate_answer(
                        question,
                        retrieved_docs
                    )

                st.markdown(answer)

                with st.expander("📚 View Sources"):

                    for doc in retrieved_docs:

                        st.write(
                            f"📄 {doc['source']} "
                            f"- Page {doc['page']}"
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as error:

                st.error(f"Error: {error}")


# =====================================
# 18. FOOTER
# =====================================

st.markdown("""
<br><br>

<div style="
    text-align:center;
    color:#94a3b8;
    padding:20px;
">

""", unsafe_allow_html=True)
