
# RAG Offline Chatbot

This is a Retrieval-Augmented Generation (RAG) offline chatbot that can answer questions based on the contents of PDF files stored locally.

##  Project Structure

```
rag_offline_chatbot/
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── pdfs/                 # Directory containing input PDF files
│   └── sample.pdf
└── models/               # Stores embedding and FAISS index files
```

##  How to Run

1. **Clone the repository**
   ```bash
   git clone "https://github.com/prathikbsgowda/rag_offline_chatbot.git"
   cd rag_offline_chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the chatbot**
   ```bash
   python app.py
   ```

4. **Ask questions about PDF content**
   You can now ask questions like:
   - "What is this document about?"
   - "Summarize the key points."
   - "Explain section 3."
   - "Who is the author?"

##  GPU Support

If a GPU is available, it will be used for faster embedding and inference. The embedding model uses `SentenceTransformer` with `device='cuda'`.

## 🛠 Requirements

- Python 3.8+
- sentence-transformers
- faiss-cpu or faiss-gpu
- PyPDF2
- streamlit

##  Notes

- Place your PDFs in the `pdfs/` directory.
- The first run will create the `models/` directory with the FAISS index.

