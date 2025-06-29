# RAG_App

A powerful **RAG (Retrieval-Augmented Generation)** application that lets you upload any PDF and ask questions about its content. This project combines **vector similarity search (FAISS)** with **LLaMA-3** (via Groq API) to generate smart, context-aware answers from the document.

> 🔗 **Live Demo on Hugging Face**:  
> [https://huggingface.co/spaces/KCU28/RAG_APPLICATION](https://huggingface.co/spaces/KCU28/RAG_APPLICATION)

---

## 💡 Features

✅ Upload any PDF  
✅ Split and embed content using SentenceTransformer  
✅ Search relevant chunks with FAISS vector similarity  
✅ Generate responses using LLaMA-3 via Groq API  
✅ Simple and clean **Gradio UI**  
✅ Deployed on Hugging Face Spaces  

---

## 🧠 How It Works

1. 📄 **Upload PDF** — Parses and extracts text using `PyPDF2`  
2. 🔍 **Chunking** — Splits text into overlapping sections  
3. 🧬 **Embeddings** — Uses `MiniLM-L6-v2` from `sentence-transformers`  
4. 📚 **Indexing** — Stores embeddings in a FAISS index  
5. ❓ **Ask a Question** — Your question is converted to an embedding  
6. 🧠 **Retrieval + Generation** — Top chunks retrieved & passed to Groq LLaMA-3 for response

---

## 🚀 Getting Started

### 🛠 Requirements

- Python 3.9+
- Groq API Key (get from [Groq Cloud](https://console.groq.com/))

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
