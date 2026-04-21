import os
import numpy as np
import requests
import chromadb
#from google import genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from bs4 import BeautifulSoup #used to parse the HTML content of the SEC filing

load_dotenv()
#client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text:str)->list[str]:
          #res = client.models.embed_content(
                    #model = "gemini-embedding-001",
                    #contents = text
          #)
          #return res.embeddings[0].values
          return model.encode(text).tolist()


#CHUNKING

def chunk_text(text:  str, chunk_size: int = 500, overlap:int =50)-> list[str]:
          words = text.split()
          chunks =[]
          start = 0
          while start< len(words):
                    end = start+ chunk_size
                    chunk = " ".join(words[start:end])
                    chunks.append(chunk)
                    start += chunk_size - overlap
          return chunks

# fetch a real SEC filing

def fetch_sec_filing(url:str)->str:
          response = requests.get(url, headers= {"User-Agent": "multi-agent-RAG-system kritikachoudhary2708@gmail.com"})
          soup = BeautifulSoup(response.content, "html.parser")
          return soup.get_text(separator=" ", strip = True)

# ingest into ChromaDB

def ingest(text: str, source: str, collection):
          chunks = chunk_text(text)
          print(f"Total chunks: {len(chunks)}")
          for i, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    collection.add(
                              ids = [f"{source}_{i}"],
                              embeddings = [embedding],
                              documents = [chunk],
                              metadatas = [{"source": source, "chunk_id": i}] #track source and chunk id
                    )
          print(f"Ingested {len(chunks)} chunks from {source}")

if __name__ == "__main__":
          
          url = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
          print("Fetching SEC filing")
          text = fetch_sec_filing(url)
          print(f"Raw text length: {len(text.split())} words")

          chroma_client = chromadb.Client()
          collection = chroma_client.create_collection(name="sec_filings")

          ingest(text, source="AAPL-10K-2023", collection=collection)

          #test
          query = "What was Apple's total revenue in 2023?"
          query_emb = get_embedding(query)
          results = collection.query(query_embeddings=[query_emb], n_results=2)

          print(f"\nQuery: {query}")
          for doc in results['documents'][0]:
                    print(f"\n→ {doc[:300]}...")  # print first 300 chars of each chunk
          
