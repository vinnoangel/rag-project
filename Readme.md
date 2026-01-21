# 🔍 RAG Document Intelligence System

A robust, local-first Retrieval-Augmented Generation (RAG) system that transforms static documents into interactive knowledge bases. This application enables users to upload PDF or TXT files and perform precise, context-grounded Q&A using state-of-the-art open-source Large Language Models (LLMs).



## 🌟 Key Features

* **Multi-Format Support:** Seamlessly parse and index `.pdf` and `.txt` files.
* **Semantic Search:** Utilizes **FAISS (HNSW)** for high-speed, high-accuracy vector retrieval.
* **Context-Grounded Answers:** Powered by **Zephyr-7B-beta**, configured with a strict system prompt to eliminate hallucinations by only answering based on provided context.
* **Intelligent Chunking:** Implements word-count-based slicing with overlapping windows to preserve semantic meaning across segments.
* **Automated Session Management:** Features a background garbage collector that purges expired session data every 2 hours to maintain system health and privacy.
* **Modern UI:** A sleek, glass-morphism themed **Gradio** web interface.

---

## 🛠️ Technical Architecture

### Core Components
* **LLM Engine:** `HuggingFaceH4/zephyr-7b-beta` via Inference API.
* **Embedding Model:** `all-MiniLM-L6-v2` (Sentence-Transformers) for converting text to 384-dimensional vectors.
* **Vector Store:** `FAISS` (Facebook AI Similarity Search) using an `HNSWFlat` index for optimized approximate nearest neighbor search.
* **Frontend:** `Gradio` with custom CSS for a premium user experience.

### Technical Workflow
1.  **Ingestion:** Text is extracted using `pypdf` and cleaned of redundant whitespace.
2.  **Chunking:** Text is split into 600-word blocks with a 100-word overlap.
3.  **Vectorization:** Chunks are embedded using `all-MiniLM-L6-v2` and L2-normalized.
4.  **Retrieval:** When a query is made, the top 3 most relevant chunks are retrieved using a cosine-similarity equivalent search.
5.  **Generation:** The LLM receives the context and query with a "Strict Assistant" persona (Temperature: 0.2).

## 📂 Project Structure

```bash
├── app.py              # Main application logic & UI
├── vector_store/       # Directory for persistent FAISS indices (auto-generated)
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```


## 🔒 Safety & Security
* **Strict Context Adherence:** The model is forbidden from using external knowledge; it only answers based on uploaded context.

* **Hallucination Mitigation:** Uses a low 0.2 temperature and limits retrieval to the top-3 most similar chunks.

* **Automated Data Purge:** A daemon thread deletes folders and cache entries older than 2 hours.

* **Session Isolation:** Every upload is isolated via a unique uuid4 session ID.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* A Hugging Face API Token (for the `InferenceClient`)

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/vinnoangel/rag-project.git](https://github.com/vinnoangel/rag-project.git)
    cd rag-document-intelligence
    ```

2.  Install dependencies:
    ```bash
    pip install huggingface_hub gradio faiss-cpu sentence-transformers pypdf
    ```

3.  Hugging Face Auth (Optional but Recommended)
    If you want to use a Hugging Face token:
    
    macOS/Linux
    ```bash
    export HF_TOKEN="your_token_here"

    ```
    
    Windows (PowerShell)
    ```bash
    $env:HF_TOKEN="your_token_here"

    ```

### Running the App
```bash
python app.py
```

The Gradio UI will start and print a local URL