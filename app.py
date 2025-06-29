

import gradio as gr
import PyPDF2
import numpy as np
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from groq import Groq
import re

class RAGSystem:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.groq_client = None
        self.setup_groq()
        self.dimension = 384
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        self.chunk_metadata = []

    def setup_groq(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            self.groq_client = Groq(api_key=api_key)
        else:
            print("Warning: GROQ_API_KEY not set.")

    def extract_text_from_pdf(self, pdf_file) -> str:
        try:
            if isinstance(pdf_file, str):
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            else:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"

    def create_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                if len(words) > overlap:
                    current_chunk = " ".join(words[-overlap:]) + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def store_in_faiss(self, embeddings: np.ndarray, chunks: List[str]):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = chunks.copy()
        self.chunk_metadata = [{"chunk_id": i, "text": chunk} for i, chunk in enumerate(chunks)]
        self.index.add(embeddings.astype('float32'))

    def search_similar_chunks(self, query: str, k: int = 3) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                results.append({
                    "chunk_id": idx,
                    "text": self.chunks[idx],
                    "score": float(score),
                    "rank": i + 1
                })
        return results

    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        if not self.groq_client:
            return "Error: GROQ_API_KEY not configured."

        context = "\n".join([chunk["text"] for chunk in context_chunks])
        prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so clearly.
Context:
{context}
Question: {query}
Answer:"""

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=1000
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def process_pdf_and_query(self, pdf_file, query, chunk_size, num_results):
        if pdf_file is None:
            return "Please upload a PDF file first.", "", ""
        if not query.strip():
            return "Please enter a question.", "", ""
        try:
            text = self.extract_text_from_pdf(pdf_file)
            if text.startswith("Error"):
                return text, "", ""
            chunks = self.create_chunks(text, chunk_size=int(chunk_size))
            if not chunks:
                return "No text chunks could be created from the PDF.", "", ""
            embeddings = self.create_embeddings(chunks)
            self.store_in_faiss(embeddings, chunks)
            relevant_chunks = self.search_similar_chunks(query, k=int(num_results))
            if not relevant_chunks:
                return "No relevant information found in the document.", "", ""
            response = self.generate_response(query, relevant_chunks)
            context_info = "**Retrieved Context:**\n\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                context_info += f"**Chunk {i} (Score: {chunk['score']:.3f}):**\n"
                context_info += f"{chunk['text'][:300]}...\n\n"
            process_info = f"""**Processing Summary:**
- PDF processed successfully
- Created {len(chunks)} text chunks
- Embedding dimension: {self.dimension}
- Retrieved {len(relevant_chunks)} relevant chunks
- Model used: llama-3.3-70b-versatile"""
            return response, context_info, process_info
        except Exception as e:
            return f"Error processing request: {str(e)}", "", ""

# Initialize
rag_system = RAGSystem()

# Gradio UI
def create_interface():
    with gr.Blocks(title="RAG Application with PDF Upload", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 📄 RAG Application with PDF Processing")
        gr.Markdown("Upload a PDF document and ask questions about its content using advanced AI retrieval.")

        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="Upload PDF Document", file_types=[".pdf"], type="filepath")
                query_input = gr.Textbox(label="Ask a Question", placeholder="What would you like to know?", lines=2)

                with gr.Row():
                    chunk_size = gr.Slider(minimum=200, maximum=1000, value=500, step=50, label="Chunk Size")
                    num_results = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Retrieved Chunks")

                submit_btn = gr.Button("Ask Question", variant="primary", size="lg")

                gr.Markdown("### Setup Instructions:\n1. Set your GROQ_API_KEY\n2. Upload a PDF\n3. Ask your question.")

            with gr.Column(scale=2):
                answer_output = gr.Textbox(label="Answer", lines=8, max_lines=20)
                with gr.Accordion("Retrieved Context", open=False):
                    context_output = gr.Markdown(label="Context")
                with gr.Accordion("Processing Info", open=False):
                    process_output = gr.Markdown(label="Processing Details")

        submit_btn.click(
            fn=rag_system.process_pdf_and_query,
            inputs=[pdf_input, query_input, chunk_size, num_results],
            outputs=[answer_output, context_output, process_output]
        )

        with gr.Accordion("Example Usage", open=False):
            gr.Markdown("""**Try questions like:**  
- "Summarize the document"  
- "What are the key findings?"  
- "Explain the methodology"  
""")
    return app

# Launch immediately for Hugging Face
app = create_interface()
app.launch()
