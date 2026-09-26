# BizIntel — Private Document Intelligence RAG Agent

BizIntel is a private document-intelligence application that lets authenticated users upload business documents, retrieve relevant context, and ask grounded questions over their own files. It uses a Lovable/React frontend, FastAPI backend, Supabase authentication/storage/database, Qdrant Cloud vector search, Jina embeddings/reranking, and Groq answer generation.

## Problem Statement

Business users often need answers from PDFs, Word files, spreadsheets, presentations, and text documents. Manual search is slow, and general chatbots may hallucinate answers from outside the uploaded file. BizIntel solves this by grounding answers in retrieved document chunks and returning a safe fallback when the answer is not available in the uploaded document.

## Design Questions

- How can document upload, retrieval, and question answering work end to end from one simple UI?
- How can retrieval stay isolated by authenticated user and selected document?
- How can answers be grounded only in uploaded document content?
- How can the app run reliably on Render without local vector database memory/storage crashes?
- How can API keys stay out of GitHub while still making deployment reproducible?

## Key Features

- Supabase email/password authentication
- User-isolated document and conversation access
- Document upload to Supabase Storage
- Document metadata and conversation persistence in Supabase Database
- Multi-format document extraction through `unstructured`
- Chunk creation for retrieved context
- Jina API embeddings
- Qdrant Cloud vector storage and retrieval
- Jina reranking
- Groq answer generation
- Strict no-context fallback when the answer is unavailable
- FastAPI health/readiness endpoints
- Docker and Render deployment support

## Tech Stack

### Frontend

- React
- Vite
- Lovable frontend structure
- Supabase JavaScript client

### Backend

- Python 3.11
- FastAPI
- Uvicorn
- Supabase Python client
- Qdrant Cloud
- Jina Embeddings API
- Jina Reranker API
- Groq-compatible chat completions
- LangChain text splitters
- `unstructured` document parsing

### Deployment

- Render Docker Web Service for backend
- Render Static Site for frontend
- Supabase Auth, Database, and Storage
- Qdrant Cloud for vector persistence

## System Architecture

```text
React/Vite Frontend
   |
   | Supabase access token
   v
FastAPI Backend on Render
   |
   +-- Supabase Auth: validate user
   +-- Supabase Storage: store uploaded files
   +-- Supabase Database: documents and conversations
   +-- Document Parser: extract text
   +-- Chunker: split text
   +-- Jina Embeddings: create vectors
   +-- Qdrant Cloud: store/search vectors
   +-- Jina Reranker: rank retrieved chunks
   +-- Groq: generate grounded answer
   v
Answer + source metadata returned to frontend
```

## RAG Flow

1. User signs in through Supabase.
2. User uploads a document.
3. Backend validates the Supabase token.
4. Backend validates file type and size.
5. File is stored in Supabase Storage.
6. Text is extracted from the uploaded file.
7. Text is chunked.
8. Chunks are embedded with Jina.
9. Vectors and metadata are stored in Qdrant Cloud.
10. User asks a question for a selected document.
11. Query is embedded with Jina.
12. Qdrant retrieves user-scoped and document-scoped chunks.
13. Jina reranks the retrieved chunks.
14. Groq generates an answer from only the retrieved context.
15. If the answer is not present, the app returns the fallback response.

## Dashboard Features

- Private authenticated workspace
- Upload document button
- Document list with processing/completed status
- Conversation list
- Selected-document Q&A
- Source-aware answer display
- User identity display
- User-isolated retrieval indicator

## Supported File Types

```text
.txt, .md, .markdown, .csv, .json, .pdf, .docx, .xlsx, .xls,
.pptx, .ppt, .doc, .html, .htm, .xml, .rtf
```

Default upload limit: `5 MB`.

## Environment Variables

Create `backend/.env` locally from `backend/.env.example`. Do not commit real `.env` files.

Required backend variables:

```env
SUPABASE_URL=your_supabase_url
AUTH_API_KEY=your_supabase_anon_or_publishable_key
DATABASE_API_KEY=your_supabase_service_role_key
STORAGE_BUCKET=documents
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL_NAME=openai/gpt-oss-20b
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

VECTOR_DB_PROVIDER=qdrant
QDRANT_CLOUD_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=enterprise_knowledge
QDRANT_VECTOR_SIZE=1024

EMBEDDING_PROVIDER=jina
JINA_API_KEY=your_jina_api_key
JINA_EMBEDDING_MODEL=jina-embeddings-v3
JINA_TIMEOUT_SECONDS=60

RERANKER_PROVIDER=jina
JINA_RERANKER_MODEL=jina-reranker-v2-base-multilingual
```

Required frontend variables:

```env
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
VITE_API_BASE_URL=your_backend_url
```

## Installation

### Backend

```powershell
cd "C:\Users\91948\repo\biz_intel_agent - Copy\backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health
```

### Frontend

```powershell
cd "C:\Users\91948\repo\biz_intel_agent - Copy\frontend"
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open:

```text
http://127.0.0.1:5173
```

## Docker

```powershell
docker build -t bizintel-api .
docker run --env-file backend/.env -p 8000:8000 bizintel-api
```

## Render Deployment

Backend:

- Runtime: Docker
- Dockerfile: `Dockerfile`
- Health check path: `/health`
- Add backend environment variables in the Render Dashboard
- Keep API keys only in Render Environment Variables

Frontend:

- Runtime: Static Site
- Build command: `cd frontend && npm ci && npm run build`
- Publish directory: `frontend/dist`
- Set `VITE_API_BASE_URL` to the backend Render URL

After moving from Chroma to Qdrant Cloud, upload documents again because old local Chroma vectors are not stored in Qdrant.

## Reproducibility

1. Clone the repository.
2. Create a Supabase project.
3. Apply `backend/schema.sql`.
4. Create a Supabase Storage bucket named `documents`.
5. Create a Qdrant Cloud cluster and API key.
6. Create a Jina API key.
7. Create a Groq API key.
8. Fill local `.env` files from examples.
9. Install backend and frontend dependencies.
10. Start backend and frontend.
11. Upload a test document.
12. Ask a question whose answer exists.
13. Ask a question whose answer does not exist and verify fallback behavior.

## Evaluation

Recommended checks:

- Upload a small `.txt` file.
- Upload a PDF or DOCX file.
- Verify chunk count is stored.
- Ask a direct factual question.
- Ask a paraphrased question.
- Ask a misspelled question.
- Ask a question not present in the document.
- Confirm the app does not answer from outside document context.
- Confirm user A cannot retrieve user B's document chunks.
- Confirm conversations are saved.
- Check Render logs for upload, embedding, Qdrant upsert, retrieval, reranking, and answer generation.

## Functionalities

- Authentication
- Document upload
- Text extraction
- Chunking
- Embedding generation
- Vector storage
- User-scoped retrieval
- Document-scoped retrieval
- Reranking
- Grounded answer generation
- Conversation persistence
- Safe fallback response
- Render-compatible deployment

## Future Improvements

- Page-level citations where extractor metadata supports it
- Background ingestion jobs for large files
- Upload progress tracking
- Richer source display in the frontend
- Hybrid keyword and vector retrieval
- Automated retrieval evaluation set
- CI checks for backend tests and frontend build
- Larger document support with job queues
- Better table-aware PDF and spreadsheet extraction

## Security Notes

- Never commit `.env` files.
- Keep `DATABASE_API_KEY` server-side only.
- Keep Qdrant, Jina, and Groq API keys in local `.env` or Render Environment Variables only.
- Frontend should only use Supabase public/anon key and backend API URL.