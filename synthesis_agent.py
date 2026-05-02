import os
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from ingestion_agent import fetch_sec_filing, chunk_text
from retrieval_agent import hybrid_search, bm25_search, dense_search
from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_embedding(text:str) -> list[float]:
          return embedder.encode(text).tolist()

# core synthesize
def synthesize(query: str, context_chunks: list[str], model: str = "llama-3.1-8b-instant") -> dict:
          context = ""
          for i, chunk in enumerate(context_chunks):
                    context += f"[Chunk {i+1}]: {chunk[:500]}\n\n"
          prompt = f""" you are financial analyst assistant. Answer the question using ONLY the provided context chunks.

          RULES:
          1. Only use information fromthe context below
          2. Always cite which chunk your answer comes from e.g. [Chunk 1]
          3. If the answer is not in the context, say "Not found in provided documents"
          4. Be precise with numbers — do not approximate or guess
          5. Keep answers concise — 1-2 sentences maximum

          Context:
          {context}

          Question: {query}
          Answer:
          """
          groq_model = model.split("/")[-1]  # strip provider prefix e.g. "groq/llama-..." → "llama-..."
          response = groq_client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role":"user", "content": prompt}],
                    temperature = 0.1
          )
          answer = response.choices[0].message.content

          #confidence score
          has_citation = any(f"[Chunk {i+1}]" in answer for i in range(len(context_chunks)))
          confidence = 0.9 if has_citation else 0.4

          return {
                    "answer": answer,
                    "confidence": confidence,
                    "chunks_used": len(context_chunks)
          }

#full rag pipeline
def rag_query(query: str, documents: list[str], collection) -> dict:
          #step 1: retrieve relevant chunks
          top_chunks = hybrid_search(query, documents, collection, top_k=3)
          #step 2: synthesize
          res = synthesize(query, top_chunks)
          return res

if __name__ == "__main__":
          # Setup
          print("Loading documents...")
          url = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
          text = fetch_sec_filing(url)
          all_chunks = chunk_text(text, chunk_size=200, overlap=20)

          #get financial chunks
          financial_chunks = [
                    c for c in all_chunks
                    if any(w in c.lower() for w in ["revenue", "net sales", "billion", "million", "fiscal 2023"])
                    and "0000320193" not in c #filter out non-apple chunks
          ]
          documents = financial_chunks[:30]
          print("\nSearching for revenue in indexed chunks:")
          for i, doc in enumerate(documents):
                    if "383" in doc or "total net sales" in doc.lower():
                              print(f"Chunk {i}: {doc[:200]}")
          #store in chromaDB
          chroma_client = chromadb.Client()
          collection = chroma_client.create_collection("rag_synthesis")
          for i, doc in enumerate(documents):
                    collection.add(
                              ids=[str(i)],
                              embeddings = [get_embedding(doc)],
                              documents= [doc],
                              metadatas=[{"chunk_index":i}]
                    )
          print(f"Indexed {len(documents)} financial chunks.")

          query = "What was Apple's total revenue in 2023?"
          top_chunks = hybrid_search(query, documents, collection, top_k=3)
          print("\nTop retrieved chunks for revenue query:")
          for i, chunk in enumerate(top_chunks):
                    print(f"\nChunk {i+1}: {chunk[:200]}")

          #Asked questions
          queries = [
                    "What was Apple's total net sales revenue in 2023?",
                    "How much did Apple spend on R&D in 2023?",
                    "What was Apple's net income in 2023?"
          ]
          for query in queries:
                    print(f"\nQuery: {query}")
                    result = rag_query(query, documents, collection)

                    print(f"\nAnswer: {result['answer']}")
                    print(f"Confidence: {result['confidence']:.2f}")
          print(f"Chunks used: {result['chunks_used']}")
