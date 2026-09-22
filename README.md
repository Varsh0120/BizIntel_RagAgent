# BizIntel

BizIntel is a multi-user document question-answering application. Authenticated users can upload documents, select an indexed document, and ask questions grounded exclusively in retrieved document content.

The application uses a React frontend, FastAPI backend, Supabase authentication and data storage, ChromaDB vector retrieval, sentence-transformer embeddings, cross-encoder reranking, and Groq for grounded answer generation.

## Features

- Email/password registration and login through Supabase Auth
- Supabase session restoration and logout
- Authenticated FastAPI endpoints
- User-scoped document and conversation access
- Document upload to Supabase Storage
- Document metadata and conversation storage in Supabase
- Text extraction from supported document formats
- Chunking and embedding generation
- Persistent ChromaDB vector indexing
- Cosine-similarity retrieval
- Cross-encoder reranking
- Document-specific question answering
- Retrieved source display
- Conversation history
- Exact no-context fallback:

  > I could not find that information in the uploaded document.

## Architecture

```text
React/Vite frontend
        |
        | Supabase access token
        v
FastAPI backend
        |
        +-- Supabase Auth
        +-- Supabase Database
        +-- Supabase Storage
        +-- ChromaDB
        +-- Sentence Transformers
        +-- Cross-Encoder Reranker
        +-- Groq Answer Generation
```

## Technology

### Frontend

- React
- Vite
- Supabase JavaScript client

### Backend

- Python 3.11
- FastAPI
- Uvicorn
- Supabase Python client
- ChromaDB
- Sentence Transformers
- Unstructured
- Groq-compatible HTTP API

## Supported document formats

- Text: `.txt`, `.md`, `.markdown`
- Structured data: `.csv`, `.json`
- PDF: `.pdf`
- Microsoft Word: `.doc`, `.docx`
- Microsoft Excel: `.xls`, `.xlsx`
- Microsoft PowerPoint: `.ppt`, `.pptx`
- Web and markup: `.html`, `.htm`, `.xml`
- Rich text: `.rtf`

The default maximum upload size is 5 MB.

## Project structure

```text
biz-intel-agent/
├── backend/
│   ├── tests/
│   ├── config.py
│   ├── main.py
│   ├── requirements.txt
│   ├── schema.sql
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── App.jsx
│   │   ├── lib/
│   │   │   ├── api.js
│   │   │   └── supabase.js
│   │   ├── index.css
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   └── .env.example
├── Dockerfile
├── render.yaml
└── README.md
```

## Prerequisites

- Python 3.11
- Node.js and npm
- Docker Desktop
- Supabase project
- Groq API key
- Git
- GitHub account

## Backend configuration

Create `backend/.env` from `backend/.env.example`.

Required secret variables:

```text
SUPABASE_URL
AUTH_API_KEY
DATABASE_API_KEY
GROQ_API_KEY
```

Important configuration:

```text
STORAGE_BUCKET=documents
GROQ_MODEL_NAME=openai/gpt-oss-20b
CHROMA_PATH=./chroma_vector_db
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Never commit `backend/.env`.

## Frontend configuration

Create `frontend/.env` from `frontend/.env.example`.

```text
VITE_SUPABASE_URL=your-supabase-project-url
VITE_SUPABASE_ANON_KEY=your-supabase-anon-key
VITE_API_BASE_URL=http://0.0.0.0:8000
```

Only the Supabase anon/publishable key belongs in frontend configuration. Never put the service-role key in the frontend.

## Supabase setup

1. Create a Supabase project.
2. Apply `backend/schema.sql` through the Supabase SQL Editor.
3. Create the Storage bucket named `documents`.
4. Configure the required Storage policies.
5. Enable the required email/password authentication settings.
6. Add local and production frontend URLs to the authentication URL configuration.

## Local backend startup

```powershell
cd backend
..\.venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend URLs:

- API: `http://127.0.0.1:8000`
- Documentation: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`
- Readiness: `http://127.0.0.1:8000/ready`

## Local frontend startup

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Tests

Run backend tests:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest -q
```

Compile backend modules:

```powershell
..\.venv\Scripts\python.exe -m py_compile main.py config.py
```

Build the frontend:

```powershell
cd frontend
npm run build
```

Audit production frontend dependencies:

```powershell
npm audit --omit=dev
```

## Docker

Build the backend image from the repository root:

```powershell
docker build --tag bizintel-api:latest .
```

Create persistent Chroma storage:

```powershell
docker volume create bizintel-chroma
```

Run the container:

```powershell
docker run --detach `
  --name bizintel-api `
  --publish 8000:8000 `
  --env-file backend\.env `
  --env PORT=8000 `
  --env CHROMA_PATH=/var/data/chroma `
  --volume bizintel-chroma:/var/data `
  bizintel-api:latest
```

Test it:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

## Render deployment

Deploy the backend as a Docker web service using:

```text
Dockerfile Path: ./Dockerfile
Docker Build Context: .
Health Check Path: /health
```

Configure:

```text
CHROMA_PATH=/var/data/chroma
```

Attach a persistent disk:

```text
Mount path: /var/data
```

Deploy the frontend as a static site:

```text
Build command: cd frontend && npm ci && npm run build
Publish directory: frontend/dist
```

Set the frontend backend URL:

```text
VITE_API_BASE_URL=https://your-backend-service.onrender.com
```

Set backend CORS:

```text
ALLOWED_ORIGINS=https://your-frontend-site.onrender.com
```

## Security

- Protected backend routes validate Supabase bearer tokens.
- Backend user identity is derived from the verified token.
- Frontend-supplied user IDs are not trusted.
- Document and conversation operations are scoped to the authenticated user.
- Chroma metadata includes both user and document ownership.
- Supabase tables use row-level security.
- The Supabase service-role key is backend-only.
- Retrieved document content is treated as untrusted data.
- The answer model receives only retrieved document context.
- Secrets and local vector data are excluded from Git and Docker builds.

## Retrieval behavior

Document chunks are embedded with:

```text
BAAI/bge-small-en-v1.5
```

ChromaDB uses cosine similarity. Candidates are filtered by authenticated user and selected document, then reranked with:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Semantic embeddings provide tolerance for paraphrases, minor spelling mistakes, and imperfect grammar. If no candidate passes the configured relevance checks, the API returns the exact fallback response.

## License

No license has been specified for this project.