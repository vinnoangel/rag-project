"""
PROJECT: RAG (Retrieval-Augmented Generation) System
DESCRIPTION: This project allows users to upload documents and query them.
It uses FAISS for vector search and Hugging Face's Zephyr-7B for generation.
"""

from huggingface_hub import InferenceClient
import gradio as gr
import faiss
import re
from pypdf import PdfReader

import uuid
import os
import json

# --- CONFIGURATION & GLOBAL INITIALIZATION ---

# Initialize the LLM client using Hugging Face's Inference API
llm = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta")

# Directory for storing vector indices and document chunks
STORAGE_DIR = "vector_store"
os.makedirs(STORAGE_DIR, exist_ok=True)

# Global state variables to hold document data and search index
embedder = None  # SentenceTransformer model instance - lazy loaded

def get_embedder_model():
    """
    Implements a Singleton/Lazy Loading pattern for the embedding model.
    The model is loaded into memory only when first needed.
    Returns:
        SentenceTransformer: Loaded embedding model instance
    """
    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        # Using a lightweight, efficient model for generating vector embeddings
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return embedder

def read_text(file_path):
    """
    Parses the uploaded file and extracts raw text.
    Supports PDF (via pypdf) and standard TXT files.
    
    Args:
        file_path (str): Path to the uploaded file
        
    Returns:
        str: Extracted and cleaned text content
    """
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        page_texts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            page_texts.append(txt)
        text = "\n".join(page_texts)
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        text = ""

    # Clean whitespace and return
    return " ".join(text.split())


def split_text(text, chunk_size=600, overlap=100):
    """
    Break text into manageable pieces for embedding and retrieval.
    
    Args:
        text (str): Input text to split
        chunk_size (int): Target number of words per chunk
        overlap (int): Number of overlapping words between consecutive chunks
        
    Returns:
        list: List of text chunks
    """
    # Use a more robust split to avoid cutting in the middle of words
    words = text.split()
    chunks = []

    # We use a word-count based approach for better semantic consistency
    # than raw character slicing
    i = 0
    while i < len(words):
        # Create a window of words
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        # Move the index forward by (chunk_size - overlap)
        i += (chunk_size - overlap)

        # Safety check to prevent infinite loops if overlap >= chunk_size
        if overlap >= chunk_size:
            i += chunk_size

    return chunks

def create_hnsw_index(embeddings):
    """
    Creates an HNSW index for lightning-fast approximate nearest neighbor search.
    
    Args:
        embeddings (numpy.ndarray): Array of document embeddings
        
    Returns:
        faiss.IndexHNSWFlat: FAISS HNSW index containing the embeddings
    """
    dimension = embeddings.shape[1]

    # M = Number of connections per node (16-64 is standard)
    # efConstruction = Depth of search during index creation (higher is better but slower)
    M = 32
    index = faiss.IndexHNSWFlat(dimension, M)
    index.hnsw.efConstruction = 80

    # Add embeddings (HNSW handles normalization internally if you use Cosine,
    # but we'll stick to Inner Product for simplicity)
    index.add(embeddings)
    return index


# Simple In-Memory Cache to prevent disk thrashing
SESSION_CACHE = {}

def process_document(file):
    """
    Main document processing pipeline: reads, chunks, embeds, and indexes a document.
    
    Args:
        file (gr.File): Gradio file object containing the uploaded document
        
    Returns:
        tuple: (session_id, status_message, chunk_count) or (None, error_message, 0)
    """
    document = read_text(file.name)

    if not document:
        return None, "Error: Document is empty.", 0

    # Use the advanced chunker now that we've preserved \n\n
    chunks = split_text(document)

    model = get_embedder_model()
    embeddings = model.encode(chunks)
    faiss.normalize_L2(embeddings)
    index = create_hnsw_index(embeddings)

    # Generate unique session ID for this document processing session
    session_id = str(uuid.uuid4())
    session_path = os.path.join(STORAGE_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)

    # Persistent Storage: Save index and chunks to disk
    faiss.write_index(index, os.path.join(session_path, "index.faiss"))
    with open(os.path.join(session_path, "chunks.json"), "w") as f:
        json.dump(chunks, f)

    # Cache the data in memory immediately for faster query response
    SESSION_CACHE[session_id] = {"index": index, "chunks": chunks}

    return session_id, f"Your document has been successfully processed.", len(chunks)

def answer(question, session_id):
    """
    Answers a question based on the indexed document content using RAG.
    
    Args:
        question (str): User's question about the document
        session_id (str): Unique identifier for the document session
        
    Returns:
        str: AI-generated answer or error message
    """
    if not session_id:
        return "Please upload a document first!"

    # Check cache first, then fall back to disk
    if session_id in SESSION_CACHE:
        data = SESSION_CACHE[session_id]
        index, chunks = data["index"], data["chunks"]
    else:
        try:
            # Load from persistent storage if not in cache
            session_path = os.path.join(STORAGE_DIR, session_id)
            index = faiss.read_index(os.path.join(session_path, "index.faiss"))
            with open(os.path.join(session_path, "chunks.json"), "r") as f:
                chunks = json.load(f)
            # Update cache for next query
            SESSION_CACHE[session_id] = {"index": index, "chunks": chunks}
        except:
            return "Session expired or files missing. Please re-upload."

    try:
        # Encode the question into embedding space
        model = get_embedder_model()
        q_embedding = model.encode([question])
        faiss.normalize_L2(q_embedding)

        # HNSW Search: Find most relevant document chunks
        index.hnsw.efSearch = 64
        scores, indices = index.search(q_embedding, k=3)

        # Filter valid indices and construct context from top chunks
        valid_indices = [i for i in indices[0] if i != -1 and i < len(chunks)]
        context = "\n\n".join([chunks[i] for i in valid_indices])

        # Generate answer using LLM with retrieved context
        response = llm.chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict retrieval assistant. Answer using ONLY the provided context. "
                        "If the answer is not in the context, say 'I do not have enough information.' "
                        "Do not use outside knowledge."
                    )
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.2  # Lower temperature for better factual accuracy
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Query Error: {str(e)}"


import time
import shutil
import threading

def start_garbage_collector(threshold_seconds=7200):
    """
    Runs in a background thread. Deletes session folders older than threshold_seconds.
    
    Args:
        threshold_seconds (int): Time in seconds after which sessions are considered expired
                                 Default: 2 hours (7200 seconds)
    """
    def cleanup_loop():
        """
        Continuous cleanup loop that runs every 10 minutes.
        """
        while True:
            now = time.time()
            if os.path.exists(STORAGE_DIR):
                for session_id in os.listdir(STORAGE_DIR):
                    path = os.path.join(STORAGE_DIR, session_id)
                    # Check the last modification time of the folder
                    if os.path.getmtime(path) < now - threshold_seconds:
                        try:
                            shutil.rmtree(path)
                            # Also remove from memory cache if present
                            if session_id in SESSION_CACHE:
                                del SESSION_CACHE[session_id]
                            print(f"Purged expired session: {session_id}")
                        except Exception as e:
                            print(f"Error purging {session_id}: {e}")

            # Sleep for 10 minutes between checks
            time.sleep(600)

    # Daemon thread ensures it closes when the main program stops
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()


# --- CSS Gradio UI ---
css = """
/* 1. Instruction Container & Equal Height Magic */
#instruction-container {
    display: flex !important;
    align-items: stretch !important;
    gap: 20px;
    background: linear-gradient(135deg, #4338ca 0%, #7e22ce 100%);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
}

#instruction-container > div {
    display: flex !important;
    flex: 1;
}

/* Glass-morphism effect for columns */
.instruction-card {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    color: white !important;
    height: 100% !important;
}

.instruction-card h3 {
    color: #f3e8ff !important;
    margin-top: 0 !important;
}

/* 2. Main App Interface Styling */
#main-container {
    background-color: #ffffff;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.action-button {
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
    border: none !important;
    transition: all 0.3s ease !important;
}

.action-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(124, 58, 237, 0.4) !important;
}
"""

# --- GRADIO INTERFACE DEFINITION ---
with gr.Blocks(css=css, title="RAG Document Intelligence") as iface:
    # State variable to track current document session
    current_session_id = gr.State(None)

    # --- STATE 1: WELCOME SCREEN ---
    with gr.Column(visible=True) as welcome_box:
        with gr.Row():
            gr.Markdown("<h1 style='text-align: center; font-size: 2.8rem;'>🔔 Welcome to RAG Intelligence</h1>")
        with gr.Row():
            gr.Markdown("<h4 style='text-align: center; color: #6b7280; margin-bottom: 20px;'>Your documents, indexed and ready for questioning.</h4>")

        with gr.Row(elem_id="instruction-container"):
            with gr.Column(elem_classes=["instruction-card"]):
                gr.Markdown("""
                ### 📋 How to Use
                1. **📄 Upload** PDF or TXT
                2. **⏳ Confirm** chunking status
                3. **❓ Ask** natural questions
                4. **💬 Get** precise answers
                """)

            with gr.Column(elem_classes=["instruction-card"]):
                gr.Markdown("""
                ### ✅ How it Works
                - **Semantic Search:** We find the 0.7+ similarity matches.
                - **Intelligent Chunking:** Optimized text segments.
                - **Grounded AI:** Powered by Zephyr-7B context.
                """)

            with gr.Column(elem_classes=["instruction-card"]):
                gr.Markdown("""
                ### ⚠️ Quick Tips
                - **Be Specific:** Ask detailed questions.
                - **Larger Docs:** Provide more context.
                - **Security:** Sessions expire after 2 hours.
                """)

        with gr.Row():
            close_btn = gr.Button("Get Started →", variant="primary", size="lg", elem_classes=["action-button"])

    # --- STATE 2: MAIN RAG INTERFACE ---
    with gr.Column(visible=False, elem_id="main-container") as main_content:
        # Header Row
        with gr.Row():
            gr.HTML("""
                <div style="text-align: center; width: 100%;">
                    <h1 style="margin-bottom: 0;">🔍 Q&A RAG System</h1>
                    <p style="color: #6b7280;">Instant insights from your uploaded data</p>
                </div>
            """)
        
        gr.HTML("<hr style='opacity: 0.1; margin: 20px 0;'>")

        with gr.Row():
            # Left Column: Upload & Control
            with gr.Column(scale=1):
                gr.Markdown("### 🛠️ Data Control")
                with gr.Group():
                    file_input = gr.File(
                        label="Source Document",
                        file_types=[".pdf", ".txt"]
                    )
                    with gr.Row():
                        status = gr.Textbox(label="Status", interactive=False, scale=2)
                        chunk_count = gr.Number(label="Chunks", interactive=False, scale=1)
            
            # Right Column: Chat & Interaction
            with gr.Column(scale=2):
                gr.Markdown("### 💬 AI Interaction")
                question_input = gr.Textbox(
                    label="Question", 
                    lines=2, 
                    placeholder="What would you like to know about the document?"
                )
                answer_btn = gr.Button("Analyze & Generate", variant="primary", elem_classes=["action-button"])
                answer_output = gr.Textbox(
                    label="AI-Generated Response", 
                    lines=8, 
                    placeholder="Results will appear here..."
                )

    # --- EVENT HANDLERS ---
    def enter_app():
        """Transition from welcome screen to main application interface."""
        return gr.update(visible=False), gr.update(visible=True)

    # Bind button click to screen transition
    close_btn.click(fn=enter_app, outputs=[welcome_box, main_content])

    # Bind file upload to document processing function
    file_input.upload(
        fn=process_document,
        inputs=file_input,
        outputs=[current_session_id, status, chunk_count]
    )

    # Bind question button to answer generation function
    answer_btn.click(
        fn=answer,
        inputs=[question_input, current_session_id], 
        outputs=answer_output
    )

# Start background garbage collection thread to clean expired sessions
start_garbage_collector()

# Launch the Gradio web server
iface.launch(share=True)