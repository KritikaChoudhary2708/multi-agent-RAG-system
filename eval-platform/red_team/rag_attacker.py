import sys
import json
import chromadb
from datetime import datetime
import os

sys.path.insert(0, '/Users/kritikachoudhary/Desktop/multi-agent-RAG-system')

from ingestion_agent import ingest, chunk_text, fetch_sec_filing
from retrieval_agent import hybrid_search
from synthesis_agent import synthesize
from prompt_library import PROMPTS

SEC_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"


def load_corpus():
    print("Fetching")
    text = fetch_sec_filing(SEC_URL)

    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection("document")
    except:
        pass
    collection = chroma_client.create_collection("document")
    ingest(text, source="apple-10k", collection=collection)
    documents = chunk_text(text)
    return documents, collection

def attack(entry, category, documents, collection):
    query = entry['prompt_text']
    if category== 'context_poisoning':
        chunks = [query]
    else:
        results = hybrid_search(query, documents, collection, top_k = 3)
        chunks = [r[0] for r in results]
    
    response = synthesize(query, chunks)
    return{
        "id": entry['id'],
        "category": category,
        "prompt_text": query,
        "expected_behaviour": entry["expected_behaviour"],
        "severity": entry["severity"],
        "response": response["answer"],
        "confidence": response["confidence"],
        "timestamp": datetime.now().isoformat()
    }

def run_all():
    documents, collection = load_corpus()
    results = []

    for category, prompts in PROMPTS.items():
        print(f"\n Running {category} ({len(prompts)} prompts):")
        for entry in prompts:
            result =  attack(entry, category, documents, collection)
            results.append(result)
    os.makedirs("../../results", exist_ok=True)
    with open("../../results/attack_results.json", "w") as f:
        json.dump(results, f, indent =2)
    print(f"\nDone. {len(results)} results → results/attack_results.json")

if __name__ == "__main__":
    run_all()