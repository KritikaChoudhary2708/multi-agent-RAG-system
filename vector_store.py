import os
import numpy as np
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key = os.getenv("GOOGLE_API_KEY"))

#--Embedding function---

def get_emb(text:str)->list[float]:
          res = client.models.embed_content(
                    model = "gemini-embedding-001",
                    contents = text
          )
          return res.embeddings[0].values

#--Vector db setup---

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name= "documents")

#--Store documents ---

def add_doc(docs: list[str]):
          for i, doc in enumerate(docs):
                    emb = get_emb(doc)
                    collection.add(
                              ids = [str(i)], #unique
                              embeddings = [emb], #vector
                              documents=[doc] #original
                    )
          print(f"Stored {len(docs)} docs!")

#---Search----

def search(query: str, top_k: int = 2)-> list[str]:
          query_emb = get_emb(query)
          res = collection.query(
                    query_embeddings = [query_emb],
                    n_results = top_k
          )
          return res['documents'][0] #returns the list of matching documents

if __name__ == "__main__":
          documents = [
                    "Apple revenue grew 12% in Q4 2023 driven by iPhone sales.",
                    "Microsoft Azure cloud revenue increased 28% year over year.",
                    "Apple's R&D spending reached $29 billion in fiscal 2023.",
                    "Google advertising revenue declined due to market slowdown.",
                    "Penguins are flightless birds native to the Southern Hemisphere."
          ]
          add_doc(documents)
          query = "What is the revenue of Apple?"
          print(f"\n Query: {query}")
          print(f"TOP RESULTS:")
          for doc in search(query=query, top_k=2):
                    print(f"- {doc}")