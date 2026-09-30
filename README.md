# LangGraph Multi-Agent RAG Assistant

A LangGraph-based multi-agent AI assistant that intelligently routes user queries between **RAG document retrieval, web search, Python execution through MCP, and direct LLM responses**, with persistent conversation memory using PostgreSQL and a FastAPI backend.

The project includes an **end-to-end RAG pipeline** that allows users to upload PDF documents, automatically chunk and embed them using a local Hugging Face embedding model, store the vectors in Pinecone, and retrieve relevant information during conversations.

## Key Features

* **LangGraph-based agent workflow** with conditional routing
* **RAG pipeline** — PDF upload → chunking → embeddings → Pinecone → retrieval
* **Local Hugging Face embeddings** using `sentence-transformers/all-MiniLM-L6-v2`
* **Pinecone vector database** with MMR-based retrieval
* **MCP integration** for Python code execution
* **Tavily web search** for real-time information
* **Persistent conversation memory** using PostgreSQL
* **Multi-turn conversations** using `thread_id`
* **Vision-enabled queries** using a vision-capable LLM
* **FastAPI REST API**
* **Pydantic request validation**
* **LangSmith observability support**
* **Bounded multi-step tool execution**
* **Bilingual responses** depending on the input language

## Architecture

```text
                         User Query
                              │
                              ▼
                       ┌─────────────┐
                       │   Router    │
                       └──────┬──────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
        Tool Required                    Direct Answer
              │
              ▼
       ┌──────────────┐
       │  LLM + Tools │
       └───────┬──────┘
               │
       ┌───────┼───────────────┐
       ▼       ▼               ▼
      RAG   Web Search      Python/MCP
       │       │               │
       └───────┴───────────────┘
               │
               ▼
        Final Answer
               │
               ▼
             END
```

## RAG Pipeline

The application provides an end-to-end document retrieval workflow:

```text
PDF Upload
    │
    ▼
PDF Text Extraction
    │
    ▼
Recursive Text Chunking
    │
    ▼
Hugging Face Embeddings
(all-MiniLM-L6-v2)
    │
    ▼
Pinecone Vector Store
    │
    ▼
MMR Retrieval
    │
    ▼
LangGraph Agent
    │
    ▼
Grounded Answer
```

### Document Upload API

```http
POST /documents/upload
```

The endpoint accepts PDF files and:

1. Extracts text from the document.
2. Splits the document into overlapping chunks.
3. Generates 384-dimensional embeddings using `all-MiniLM-L6-v2`.
4. Stores the vectors and document metadata in Pinecone.
5. Makes the document available to the RAG agent.

Example response:

```json
{
  "message": "Document indexed successfully.",
  "filename": "project-documentation.pdf",
  "chunks_indexed": 12
}
```

## Available Tools

| Tool        | Technology                         | Purpose                                           |
| ----------- | ---------------------------------- | ------------------------------------------------- |
| RAG         | Pinecone + Hugging Face Embeddings | Retrieve relevant content from uploaded documents |
| Web Search  | Tavily                             | Retrieve current information from the web         |
| Python Tool | MCP                                | Execute Python code through an MCP server         |
| Direct LLM  | NVIDIA NIM                         | Answer general queries without external tools     |

## Project Structure

```text
langgraph-multi-agent/
│
├── main.py
│   └── FastAPI application, chat API and document upload API
│
├── graph.py
│   └── LangGraph workflow assembly
│
├── state.py
│   └── Graph state and router decision schemas
│
├── llms.py
│   └── LLM and embedding model configuration
│
├── config.py
│   └── Environment configuration and validation
│
├── document_ingestion.py
│   └── PDF loading, chunking and Pinecone indexing
│
├── nodes/
│   ├── router.py
│   ├── llm_tool.py
│   ├── conditions.py
│   └── answer.py
│
├── tools/
│   ├── rag.py
│   ├── web_search.py
│   └── mcp_client.py
│
├── docs/
│   ├── architecture.md
│   └── api_reference.md
│
├── tests/
│   └── test_agent.py
│
├── .env.example
├── requirements.txt
├── LICENSE
└── README.md
```

## Tech Stack

| Layer           | Technology                         |
| --------------- | ---------------------------------- |
| Agent Framework | LangGraph                          |
| LLM             | NVIDIA NIM                         |
| LLM Integration | LangChain                          |
| Embeddings      | Hugging Face Sentence Transformers |
| Vector Database | Pinecone                           |
| Retrieval       | MMR                                |
| Web Search      | Tavily                             |
| Tool Protocol   | MCP                                |
| Backend         | FastAPI                            |
| Validation      | Pydantic                           |
| Memory          | PostgreSQL                         |
| Observability   | LangSmith                          |
| Language        | Python 3.11                        |

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd langgraph-multi-agent
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file based on `.env.example`.

```env
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=openai/gpt-oss-20b

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=rag-documents

TAVILY_API_KEY=your_tavily_api_key

DATABASE_URL=your_postgresql_connection_string

MCP_SERVER_URL=your_mcp_server_url
```

Never commit the `.env` file or API keys to GitHub.

### 5. Run the API

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

### Chat

```http
POST /chat
```

Request:

```json
{
  "query": "What is machine learning?",
  "thread_id": "1"
}
```

### Upload Document

```http
POST /documents/upload
```

Upload a PDF through the Swagger UI or any API client.

After indexing the document, ask questions related to its contents through `/chat`.

## Configuration

| Variable               | Required | Purpose                             |
| ---------------------- | -------- | ----------------------------------- |
| `NVIDIA_API_KEY`       | Yes      | NVIDIA NIM LLM inference            |
| `PINECONE_API_KEY`     | Yes      | Pinecone authentication             |
| `PINECONE_INDEX`       | Yes      | Pinecone vector index               |
| `TAVILY_API_KEY`       | Yes      | Web search                          |
| `DATABASE_URL`         | Yes      | PostgreSQL conversation memory      |
| `MCP_SERVER_URL`       | Optional | MCP server endpoint                 |
| `LANGCHAIN_API_KEY`    | Optional | LangSmith tracing                   |
| `LANGCHAIN_PROJECT`    | Optional | LangSmith project                   |
| `LANGCHAIN_TRACING_V2` | Optional | Enable LangSmith tracing            |
| `MAX_ITERATIONS`       | Optional | Maximum agent tool iterations       |
| `RETRIEVER_K`          | Optional | Number of retrieved documents       |
| `RETRIEVER_FETCH_K`    | Optional | Candidate documents fetched for MMR |

## Example Workflow

```text
1. Upload a PDF through /documents/upload
              ↓
2. PDF is split into chunks
              ↓
3. Chunks are converted into embeddings
              ↓
4. Embeddings are stored in Pinecone
              ↓
5. User asks a question through /chat
              ↓
6. LangGraph determines whether RAG is required
              ↓
7. Relevant document chunks are retrieved
              ↓
8. LLM generates the final response
```

## License

This project is based on the original work by **Mohammad Arbaz (Arbaz / Pokemon455)** and is distributed under the MIT License.

The original MIT License and copyright notice are preserved in the `LICENSE` file.

## Acknowledgements

* LangGraph
* LangChain
* Pinecone
* Hugging Face Sentence Transformers
* NVIDIA NIM
* Tavily
* FastMCP / MCP
* FastAPI
