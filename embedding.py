from google import genai
import os 
from dotenv import load_dotenv
import numpy as np

load_dotenv() #loads the environment variables from the .env file
client= genai.Client(api_key = os.getenv("GOOGLE_API_KEY"))
for m in client.models.list():
    if 'embed' in m.name.lower():
        print(m.name)

def get_embedding(text:str)->list[float]:
          res = client.models.embed_content(
                    model = "gemini-embedding-001",
                    contents = text
          )
          return res.embeddings[0].values

def cosine_similarity(vec1: list, vec2: list)->float:
          a = np.array(vec1)
          b = np.array(vec2)
          return float(np.dot(a,b)/ (np.linalg.norm(a)*np.linalg.norm(b))) #linalg.norm is used to calculate the magnitude of the vector

if __name__ == "__main__":

          sentence1 = "Apple reported strong revenue growth in Q4"
          sentence2 = "The company saw increased earnings this quarter"
          sentence3 = "Penguins live in Antarctica"

          emb1 = get_embedding(sentence1)
          emb2 = get_embedding(sentence2)
          emb3 = get_embedding(sentence3)

          print(f"Vector size: ", len(emb1))
          print(f"Similarity between sentence 1 and 2: ", cosine_similarity(emb1, emb2))
          print(f"Similarity between sentence 1 and 3: ", cosine_similarity(emb1, emb3))
