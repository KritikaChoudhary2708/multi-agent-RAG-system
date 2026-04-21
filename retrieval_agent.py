import os
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text:str)->list[float]:
          return model.encode(text).tolist()

# BM25 Search

def bm25_search(query: str, documents: list[str], top_k: int=5)-> list[tuple]: #keyword based search
          token_docs = [doc.lower().split() for doc in documents]
          token_query = query.lower().split()

          bm25 = BM25Okapi(token_docs)
          scores = bm25.get_scores(token_query)

          rank = sorted(enumerate(scores), key = lambda x: x[1], reverse=True)
          return rank[:top_k]

# Dense search

def dense_search(query: str, collection, top_k: int=5)-> list[tuple]: #semantic search
          query_emb = get_embedding(query)
          res = collection.query(
                    query_embeddings = [query_emb],
                    n_results = top_k,
                    include=['documents', 'distances']
          )
          docs = res['documents'][0]
          dists = res['distances'][0]
          return list(zip(docs, dists)) #zip: combines the two lists

#Reciprocal Rank Fusion

def rrf(bm25_res, den_res, documents, k:int = 60) -> list[tuple]: #combines the results of both the searches
          scores={}
          for rank, (doc_idx, _) in enumerate(bm25_res): # _ is used to ignore the score from bm25_res
                    doc = documents[doc_idx]
                    scores[doc] =  scores.get(doc, 0)+ 1 / (rank+k)
          for rank, (doc, _) in enumerate(den_res):
                    scores[doc] = scores.get(doc, 0) + 1 / (rank+k)
          
          fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
          return [doc for doc, _ in fused]

#hybrid

def hybrid_search(query: str, documents: list[str], collection, top_k: int = 3)-> list[tuple]:
          bm25_res = bm25_search(query, documents, top_k =5)
          den_res = dense_search(query, collection, top_k =5)
          fused = rrf(bm25_res, den_res, documents, k=60)
          return fused[:top_k]

def is_clean_chunk(text: str) -> bool: #filters out the XBRL metadata chunks
    """Filter out XBRL metadata chunks — they're noise, not readable text."""
    xbrl_signals = ["0000320193", "us-gaap:", "iso4217:", "xbrli:"]
    return not any(signal in text for signal in xbrl_signals)

if __name__ == "__main__":
          from ingestion_agent import fetch_sec_filing, chunk_text
          print("Fetching SEC filing")
          url = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
          text = fetch_sec_filing(url)
         #instead of selecting random chunks, we will find chunks which will have revenue/financial data
          all_chunks = chunk_text(text)
          fin_chunks =[
                    c for c in all_chunks
                    if any(word in c.lower() for word in ["revenue", "net sales", "earnings", "billion", "million", "fiscal"])

          ]
          docs = fin_chunks[:30]
          print(f"Selected {len(docs)} financial chunks")


          # store in chromadb
          chroma_client = chromadb.Client()
          collection = chroma_client.create_collection("sec_hybrid")
          for i, doc in enumerate(docs):
                    doc = str(doc)
                    if not is_clean_chunk(doc):
                              continue
                    collection.add(
                              ids=[str(i)],
                              embeddings=[get_embedding(doc)],
                              documents=[doc],
                              metadatas=[{"chunk_index":i}]
                    )
          print(f'Stored {len(docs)} chunks')
          query ="What was Apple total revenue in 2023?"
          print(f"\n Query:{query}")
          print("BM25---> ")
          for idx, score in bm25_search(query, docs, top_k=3):
                    print(f"Score: {score:.4f} | Doc: {docs[idx][:200]}...")
          
          print("\nDense---> ")
          for doc, dist in dense_search(query, collection, top_k=3):
                    print(f"Distance: {dist:.4f} | Doc: {doc[:200]}...")
          
          print("\nHybrid---> ")
          for doc in hybrid_search(query, docs, collection, top_k=3):
                    print(f"Doc: {doc[:200]}...")


          

