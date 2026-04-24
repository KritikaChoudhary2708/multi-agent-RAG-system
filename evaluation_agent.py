import os
import chromadb
from ragas import evaluate
from datasets import Dataset
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

embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("GROQ KEY:", os.getenv("GROK_API_KEY")[:10] if os.getenv("GROK_API_KEY") else "NOT FOUND")

# --- Setup RAG pipeline (same as before) ---
def setup_pipeline():
    url = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    text = fetch_sec_filing(url)
    all_chunks = chunk_text(text, chunk_size=500, overlap=50)

    def is_financial_chunk(text: str) -> bool:
        if "0000320193" in text: return False
        if len(text.split()) < 30: return False
        has_numbers = "$" in text or "%" in text
        financial_words = ["revenue", "net sales", "net income", "research",
                           "earnings", "gross margin", "billion", "million"]
        return has_numbers and any(w in text.lower() for w in financial_words)

    documents = [c for c in all_chunks[30:70] if is_financial_chunk(c)][:30]

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("ragas_eval")
    for i, doc in enumerate(documents):
        collection.add(
            ids=[str(i)],
            embeddings=[get_embedding(doc)],
            documents=[doc],
            metadatas=[{"chunk_index": i}]
        )
    return documents, collection


# Run RAG on each golden question
def run_rag_on_dataset(golden_dataset, documents, collection):
          questions = []
          answers = []
          contexts=[]
          ground_truth = []

          for item in golden_dataset:
                    ques = item["question"]
                    gt = item["ground_truth"]

                    #retrieve
                    top_chunks = hybrid_search(ques, documents, collection, top_k =3)

                    #synthesize
                    res = synthesize(ques, top_chunks)
                    questions.append(ques)
                    answers.append(res["answer"])
                    contexts.append(top_chunks) #list of retrieved chunks
                    ground_truth.append(gt)
                    print(f"Q: {questions}")
                    print(f"A: {answers}")
                    print(f"Confindence: {res['confidence']}")

                    return {
                              "question": questions,
                              "answer" : answers,
                              "contexts": contexts,
                              "ground_truth" :ground_truth
                    }


#score with ragas
def score_with_ragas(ragas_data: dict):
          dataset = Dataset.from_dict(ragas_data)

          #LLM for RAGAS scoring (GROQ)
          llm = ChatGroq(
                    model = "llama-3.1-8b-instant",
                    api_key=os.getenv("GROK_API_KEY")
          )
          #embedding for RAGAS
          embeddings = HuggingFaceEmbeddings(
                    model_name = "sentence-transformers/all-MiniLM-L6-v2"
          )
          res = evaluate(
                    dataset=dataset,
                    metrics = [faithfulness, answer_relevancy, context_recall, context_precision],
                    llm = llm,
                    embeddings= embeddings
          )
          return res

if __name__ == "__main__":
          print("Setting up pipeline...")
          documents, collection = setup_pipeline()
          print(f"Indexed {len(documents)} chunks. \n")

          print("Running RAG on golden dataset...")
          ragas_data = run_rag_on_dataset(golden_dataset, documents, collection)

          print("\n Scoring with RAGAS...")
          scores = score_with_ragas(ragas_data)
          print(f"\nFaithfulness:     {scores['faithfulness']}")
          print(f"Answer Relevancy: {scores['answer_relevancy']}")
          print(f"Context Recall:   {scores['context_recall']}")
          print(f"Context Precision:{scores['context_precision']}")

