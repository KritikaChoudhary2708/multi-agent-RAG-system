# Goal
Multi-agent RAG over a real document corpus (SEC filings, Indian court judgments, or research papers) with full
evaluation pipeline running in CI/CD.

# Dataset / Data Source
SEC EDGAR (free), arXiv papers, Indian Kanoon (legal docs), or synthetic enterprise docs via Faker

# Architecture & Tech Stack
• Ingestion agent: semantic chunking, metadata extraction, embedding
• Retrieval agent: hybrid search (BM25 + dense), reranking, context compression
• Synthesis agent: answer generation with citations + confidence scores
• Evaluation agent: RAGAS scores auto-run on every PR as a deployment gate
• Monitoring: Langfuse dashboard tracking latency, cost-per-query, hallucination rate

# Key Technical Challenges
• Multi-hop questions requiring chained retrievals
• Query decomposition for complex questions
• Reducing hallucination rate below 5% on eval set

# Evaluation metrics that matter to recruiters: 
RAGAS faithfulness >0.85, answer relevancy >0.80, latency p95 <3s, cost-per-query tracked, evals passing in CI

# Advanced Extensions

1. Add a Critic agent that flags low-confidence answers and routes to human review
2. Implement HyDE (Hypothetical Document Embeddings) and A/B test vs. naive retrieval
3. Add streaming with SSE, deploy on Railway or Render with a real domain
4. Build a nightly eval regression suite that emails you if quality drops