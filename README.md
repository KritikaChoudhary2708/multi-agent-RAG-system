# Multi-Agent RAG System

A modular, multi-agent Retrieval-Augmented Generation (RAG) system designed to fetch, process, index, and intelligently query complex financial documents like SEC filings (e.g., 10-K reports). The system is built using Python, ChromaDB for vector storage, Sentence Transformers for embeddings, and the Groq API for fast LLM inference.

## Features

- **Multi-Hop Query Pipelines**: Decomposes complex queries into simpler sub-questions, retrieves information for each individually, and synthesizes them into a combined final answer.
- **Automated Data Ingestion**: Fetches SEC filings directly via HTTP, parses HTML, and chunks the text while cleaning out metadata noise (e.g., XBRL tags).
- **Hybrid Search Retrieval**: Combines keyword-based search (BM25) and semantic search (Dense Embeddings) using Reciprocal Rank Fusion (RRF) to retrieve the most relevant context.
- **LLM Synthesis via Groq**: Uses Llama-3 models via the Groq API to synthesize answers based _strictly_ on the retrieved context chunks.
- **Grounded Answers**: Generated responses include explicit citations (e.g., `[Chunk 1]`) and an estimated confidence score.

## Architecture

The system is decomposed into specialized agents:

1. **`ingestion_agent.py`**: 
   - Downloads HTML from a specified URL (e.g., SEC EDGAR).
   - Generates chunks of text with specified size and overlap.
   - Encodes chunks into dense vectors using `sentence-transformers` (`all-MiniLM-L6-v2`).
   - Stores the documents and embeddings into ChromaDB.

2. **`retrieval_agent.py`**:
   - Implements BM25 for sparse/keyword search.
   - Implements Dense Search using ChromaDB.
   - Merges results using Reciprocal Rank Fusion (RRF) for a robust Hybrid Search.
   - Contains logic to filter out unreadable XBRL data chunks.

3. **`synthesis_agent.py`**:
   - Orchestrates the full pipeline.
   - Constructs strict prompts instructing the LLM to only use provided context.
   - Calls the `llama-3.1-8b-instant` model via the Groq API.
   - Calculates a confidence score based on the presence of valid citations in the output.

4. **`query_decomposition.py`**:
   - Implements a multi-hop query workflow.
   - Decomposes complex multi-part queries into standalone sub-questions.
   - Retrieves and answers each sub-question independently using the hybrid search pipeline.
   - Synthesizes a final, comprehensive answer by combining the sub-answers intelligently.

5. **`vector_store.py` / `embedding.py`**:
   - Utilities and standalone examples for interacting with Vector Databases and alternative embedding setups (e.g., Gemini API).

## Prerequisites

Before running the project, you need to set up your environment variables. 
Create a `.env` file in the root directory and add the following keys:

```env
GROK_API_KEY=your_groq_api_key_here
# Optional: Add Google API key if using Gemini for embeddings in alternative scripts
GOOGLE_API_KEY=your_google_api_key_here
```

## Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the necessary dependencies (ensure these are in your environment):
   ```bash
   pip install chromadb sentence-transformers requests beautifulsoup4 python-dotenv rank_bm25 groq numpy google-genai
   ```

## Usage

You can run individual components to see them in action.

**To run the end-to-end synthesis pipeline:**
```bash
python synthesis_agent.py
```
This will fetch an Apple 10-K SEC filing, process financial chunks, index them, retrieve context for sample queries (e.g., "What was Apple's total revenue in 2023?"), and output the LLM's synthesized response with citations.

**To test just the ingestion module:**
```bash
python ingestion_agent.py
```

**To test the retrieval components (BM25 vs Dense vs Hybrid):**
```bash
python retrieval_agent.py
```

**To run the complex multi-hop decomposition pipeline:**
```bash
python query_decomposition.py
```
