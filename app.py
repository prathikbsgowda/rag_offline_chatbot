import os
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline,AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load PDF and split to chunks
def load_pdf_chunks(pdf_path, chunk_size=500):
    reader = PyPDF2.PdfReader(open(pdf_path, 'rb'))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + " "
    # Chunking
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks

# Embed the chunks
def embed_chunks(chunks, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings, model

# Create FAISS index
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Search FAISS
def retrieve_top_chunks(query, embedder, chunks, index, k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), k)
    results = [chunks[i] for i in I[0]]
    return results

# Load local LLM (Mistral 7B via HuggingFace)
def load_llm():
    print("🤖 Loading FLAN-T5 model (public, fast, no login)...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    return pipe

# Generate answer using context
def generate_answer(question, context, generator):
    prompt = f"""Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""
    output = generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].split("Answer:")[-1].strip()

# Main
if __name__ == "__main__":
    print("🔍 Loading PDF...")
    chunks = load_pdf_chunks("pdfs/sample.pdf")

    print("🔗 Creating embeddings...")
    embeddings, embedder = embed_chunks(chunks)
    embeddings = np.array(embeddings)

    print("📚 Indexing with FAISS...")
    index = create_faiss_index(embeddings)

    print("🤖 Loading local LLM (may take a few minutes)...")
    generator = load_llm()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        top_chunks = retrieve_top_chunks(query, embedder, chunks, index)
        context = "\n".join(top_chunks)
        answer = generate_answer(query, context, generator)
        print("\n🧠 Answer:", answer)
