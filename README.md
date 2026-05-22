<div align="center">

# 🛡️ CyberGraph RAG

### Agentic Cybersecurity Knowledge Assistant

**CyberGraph RAG** is an Agentic Retrieval-Augmented Generation system designed for cybersecurity, cloud, AI, and technical documents.  
It combines document ingestion, parent-child chunking, vector search, local LLM inference, query rewriting, relevance grading, hallucination checking, and source-grounded answer generation.

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC244C?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=for-the-badge)

</div>

---

## 🚀 Overview

CyberGraph RAG is a full-stack **Agentic RAG assistant** that allows users to upload documents and ask grounded questions from those documents.

Unlike a normal RAG chatbot, this system does not simply retrieve chunks and generate an answer.  
It performs multiple agentic reasoning steps before producing the final response.

The system can:

- Upload and process PDF, TXT, and Markdown documents
- Convert PDFs into clean Markdown text
- Split documents using parent-child chunking
- Store child chunks in Qdrant vector database
- Retrieve larger parent contexts for better answer quality
- Rewrite user questions into better search queries
- Grade retrieved context relevance
- Generate grounded answers using a local Ollama LLM
- Check whether answers are supported by retrieved context
- Regenerate safer answers if unsupported claims are detected
- Display sources, relevance grades, and hallucination checks in a Streamlit UI

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📄 Document Upload | Supports PDF, TXT, Markdown, and `.md` files |
| 🧹 Text Cleaning | Removes noisy spaces, blank lines, and formatting issues |
| 🧩 Parent-Child Chunking | Uses small chunks for search and larger chunks for answer context |
| 🧠 Local LLM | Uses Ollama for local/offline answer generation |
| 🔎 Vector Search | Uses Qdrant with HuggingFace sentence-transformer embeddings |
| 🔁 Query Rewriting | Rewrites vague user questions into better retrieval queries |
| ✅ Relevance Grading | Filters out weak or unrelated retrieved contexts |
| 🛡️ Hallucination Checking | Checks if the answer is supported by retrieved documents |
| ♻️ Safer Regeneration | Regenerates answer when unsupported claims are detected |
| 🌐 FastAPI Backend | Provides clean REST API and Swagger documentation |
| 🎨 Streamlit Frontend | Provides an interactive UI for upload and chat |

---

## 🧠 Why This Project Matters

Most basic RAG systems follow this simple pipeline:

```text
User Question → Vector Search → LLM Answer

User Question
      ↓
Query Rewriting Agent
      ↓
Vector Search on Child Chunks
      ↓
Parent Context Retrieval
      ↓
Document Relevance Grader
      ↓
Grounded Answer Generator
      ↓
Hallucination Checker
      ↓
Optional Safer Regeneration
      ↓
Final Answer with Sources

```


## 🏗️ System Architecture

CyberGraph RAG
│
├── Document Ingestion
│   ├── PDF/TXT/Markdown Upload
│   ├── PDF to Markdown Conversion
│   ├── Text Cleaning
│   └── Parent-Child Chunking
│
├── Storage Layer
│   ├── Parent Chunks Stored Locally
│   └── Child Chunks Stored in Qdrant
│
├── Retrieval Layer
│   ├── Query Rewriting
│   ├── Child Chunk Vector Search
│   ├── Parent Context Loading
│   └── Context Formatting
│
├── Agentic Reasoning Layer
│   ├── Relevance Grading
│   ├── Grounded Answer Generation
│   ├── Hallucination Checking
│   └── Safer Answer Regeneration
│
├── Backend
│   └── FastAPI REST API
│
└── Frontend
    └── Streamlit Web Interface

 ## 🛠️ Tech Stack

 | Layer            | Technology                        |
| ---------------- | --------------------------------- |
| Language         | Python                            |
| Backend          | FastAPI                           |
| Frontend         | Streamlit                         |
| Local LLM        | Ollama                            |
| Embeddings       | HuggingFace Sentence Transformers |
| Vector Database  | Qdrant                            |
| RAG Framework    | LangChain                         |
| Document Parsing | PyMuPDF / pymupdf4llm             |
| API Docs         | Swagger UI                        |
| Environment      | Conda / venv                      |

## 📂 Project Structure

cybergraph-rag-assistant/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── chat.py
│   │   │   └── documents.py
│   │   │
│   │   ├── services/
│   │   │   ├── chat_service.py
│   │   │   ├── document_service.py
│   │   │   ├── hallucination_checker_service.py
│   │   │   ├── llm_service.py
│   │   │   ├── parent_store_service.py
│   │   │   ├── query_rewriter_service.py
│   │   │   ├── relevance_grader_service.py
│   │   │   ├── retrieval_service.py
│   │   │   └── vector_store_service.py
│   │   │
│   │   ├── utils/
│   │   │   ├── chunker.py
│   │   │   ├── file_loader.py
│   │   │   └── text_cleaner.py
│   │   │
│   │   ├── models/
│   │   │   └── schemas.py
│   │   │
│   │   ├── data/
│   │   │   ├── uploads/
│   │   │   ├── markdown/
│   │   │   ├── vector_db/
│   │   │   └── parent_store/
│   │   │
│   │   ├── config.py
│   │   └── main.py
│   │
│   └── requirements.txt
│
├── frontend/
│   └── streamlit_app.py
│
├── assets/
│   └── screenshots/
│
├── .env.example
├── .gitignore
└── README.md

## 🎯 Use Cases

CyberGraph RAG can be adapted for:

- Cybersecurity report Q&A
- Cloud documentation assistant
- Research paper assistant
- Resume and portfolio knowledge assistant
- Technical knowledge base search
- Security policy and compliance document review
- Internal enterprise documentation assistant

## 🧑‍💻 What I Built

This project demonstrates practical AI engineering skills:

- End-to-end RAG system design
- Backend API development with FastAPI
- Frontend development with Streamlit
- Local LLM integration using Ollama
- Vector database integration with Qdrant
- Embedding-based semantic search
- Parent-child retrieval architecture
- Agentic query rewriting
- LLM-based relevance grading
- Hallucination detection and safer regeneration
- Source-grounded answer generation

## 🔮 Future Improvements
- Add LangGraph workflow visualization
- Add Docker and Docker Compose support
- Add user authentication
- Add document deletion and re-indexing
- Add conversation memory
- Add hybrid dense + sparse retrieval
- Add DOCX and CSV support
- Add RAG evaluation metrics
- Add cloud deployment
- Add role-based document access
- Add exportable chat history

<div align="center">
⭐ If you find this project useful, consider starring the repository.

CyberGraph RAG — Agentic AI for trusted technical knowledge retrieval

</div> ```