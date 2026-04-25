import os
import json
import chromadb
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from ingestion_agent import fetch_sec_filing, chunk_text
from retrieval_agent import hybrid_search
from synthesis_agent import synthesize, get_embedding
from golden_dataset import golden_dataset
from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def setup_pipeline():
          url = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
          text = fetch_sec_filing(url)
          all_chunks = chunk_text(text, chunk_size=500, overlap=50)

          def is_financial_chunk(text):
                    if "0000320193" in text: return False
                    if len(text.split()) < 30: return False

                    return ("$" in text or "%" in text) and any(
                              w in text.lower() for w in ["revenue", "net sales", "net income",
            "research", "earnings", "gross margin", "billion", "million"]
                    )
          
          documents = [c for c in all_chunks[30:70] if is_financial_chunk(c)][:30]

          chroma_client = chromadb.Client()
          collection = chroma_client.create_collection("ci_eval")

          for i, doc in enumerate(documents):
                    collection.add(
                              ids = [str(i)],
                              embeddings = [get_embedding(doc)],
                              documents=[doc],
                              metadatas=[{"chunk_index":i}]
                    )
          return documents, collection


def run_evaluation():
          print("=== CI Evaluation v2 ===") 
          documents, collection = setup_pipeline()
          questions, answers, contexts, ground_truths = [], [], [], []

          for item in golden_dataset:
                    top_chunks = hybrid_search(item["question"], documents, collection, top_k=3)
                    res = synthesize(item["question"], top_chunks)
                    questions.append(item["question"])
                    answers.append(res["answer"])
                    contexts.append(top_chunks)
                    ground_truths.append(item["ground_truth"])
          
          dataset = Dataset.from_dict({
                    "question": questions,
                    "answer": answers,
                    "contexts": contexts,
                    "ground_truth": ground_truths
          })

          llm = ChatGroq(
                    model = "llama-3.3-70b-versatile",
                    api_key = os.getenv("GROK_API_KEY"),
                    n=1,                  
                    max_tokens=4096      
          )
          embeddings = HuggingFaceEmbeddings(
                    model_name = "sentence-transformers/all-MiniLM-L6-v2"
          )
          
          scores = evaluate(
                    dataset = dataset,
                    metrics=[
                              faithfulness,
                              answer_relevancy,
                              context_recall,
                              context_precision
                    ],
                    llm = llm,
                    embeddings = embeddings
          )

          res ={
                    "faithfulness": float(scores["faithfulness"][0]),
                    "answer_relevancy": float(scores["answer_relevancy"][0]),
                    "context_recall": float(scores["context_recall"][0]),
                    "context_precision": float(scores["context_precision"][0]),
          }

          with open("eval_scores.json", "w") as f:
                    json.dump(res, f, indent=2)
          print(json.dumps(res, indent=2))
          return res


if __name__ == "__main__":
          run_evaluation()
                    