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
                    api_key = os.getenv("GROQ_API_KEY"),
                    n=1,                  
                    max_retries=3,
                    timeout=60,            
                    request_timeout=60,   
          )
          ragas_llm = LangchainLLMWrapper(groq_llm)
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
                    llm = ragas_llm,
                    embeddings = embeddings
          )
          result = run_eval_with_retry(
                    dataset,
                    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
                    llm=ragas_llm,
                    embeddings=embeddings
          )
          if result is None:
                print("Evaluation failed after all retries")
                exit(1)

          # Handle NaN scores
          df = result.to_pandas()
          df = df.fillna(0)

          scores = {
                "faithfulness":      float(df["faithfulness"].mean()),
                "answer_relevancy":  float(df["answer_relevancy"].mean()),
                "context_recall":    float(df["context_recall"].mean()),
                "context_precision": float(df["context_precision"].mean())
          }

          with open("eval_scores.json", "w") as f:
                    json.dump(res, f, indent=2)
          print(json.dumps(res, indent=2))
          return res

def run_eval_with_retry(dataset, metrics, llm, embeddings, retries=3):
    for attempt in range(retries):
        try:
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=False,   # KEY: don't crash on single failures
            )
            return result
        except Exception as e:
            print(f"Eval attempt {attempt+1} failed: {e}")
            time.sleep(10 * (attempt + 1))   # backoff: 10s, 20s, 30s
    return None

if __name__ == "__main__":
          run_evaluation()
                    