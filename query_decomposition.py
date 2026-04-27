import os
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from ingestion_agent import fetch_sec_filing, chunk_text
from retrieval_agent import hybrid_search
from synthesis_agent import synthesize, get_embedding
from dotenv import load_dotenv

load_dotenv()
embedder =  SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=os.getenv("GROK_API_KEY"))

#decompose query
def decompose_query(query:str)->dict:
          prompt = f'''
          Break the following questions into simple, independent sub-questions.
          Each sub-question should be answerable on its own from a financial document.
          Return ONLY the sub-questions, one per line, no numbering, no explaination.
          If the question is already simple, return it as in on one line.

          Question: {query}

          Sub-questions:
          '''  
          response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content" : prompt}],
                    temperature=0.1
          )
          raw = response.choices[0].message.content
          sub_questions = [q.strip() for q in raw.strip().split('\n') if q.strip()]
          return sub_questions

#retrieve for each sub-question
def ans_sub_ques(sub_ques: list[str], docs: list[str], collection) -> list[dict]:
          res =[]
          for sub_q in sub_ques:
                    print(f"Retrieving for:{sub_q}")
                    top_chunks = hybrid_search(sub_q, docs, collection, top_k=2)
                    result = synthesize(sub_q, top_chunks)
                    res.append({
                              "sub_question": sub_q,
                              "answer": result["answer"],
                              "confidence": result["confidence"]
                    })
          return res

# combine sub ans into final ans
def combine_ans(o_query: str, sub_res : list[dict])-> str:
          sub_ans_text = ""
          for r in sub_res:
                    sub_ans_text +=f"Q:{r['sub_question']}\n A:{r['answer']}\n Confidence:{r['confidence']}\n\n"
          prompt= f"""
          You have answers to several sub-questions. 
          Combine them into one coherent, concise answer to the original question.
          Only use the information provided — do not add anything new.

          Original question: {o_query}

          Sub-answers:
          {sub_ans_text}

          Final combined answer:
          """
          response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content" : prompt}],
                    temperature=0.1
          )
          return response.choices[0].message.content

#full multi hop pipeline

def multi_hop_query(query: str, documents:list[str], collection)-> dict:
          # decompose-> retrieve -> combine
          print(f"\nOriginal query:{query}")
          
          #step1 decompose
          sub_questions = decompose_query(query)
          print(f"\nDecomposed into {len(sub_questions)} sub-ques:")
          for q in sub_questions:
                    print(f"-{q}")
          
          #step2 retrieve for each sub-question
          sub_res = ans_sub_ques(sub_questions, documents, collection)

          #step 3 combine
          final_ans = combine_ans(query, sub_res)
          
          return {
                    "original_query": query,
                    "sub_ques": sub_questions,
                    "sub_results": sub_res,
                    "final_answer": final_ans
          }

#test
def is_financial_chunk(text: str) -> bool:
    # Skip junk
    if "0000320193" in text: return False
    if len(text.split()) < 30: return False

    # Must contain actual dollar figures or percentages
    has_numbers = "$" in text or "%" in text
    
    # Must contain financial keywords
    financial_words = ["revenue", "net sales", "net income", "r&d",
                   "research and development", "research", "earnings", 
                   "gross margin", "operating income", "billion", "million"]
    has_keywords = any(w in text.lower() for w in financial_words)

    return has_numbers and has_keywords  # BOTH must be true

if __name__ == "__main__":
    print("Loading documents...")
    url = "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    text = fetch_sec_filing(url)
    all_chunks = chunk_text(text, chunk_size=500, overlap=50)
    all_chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"Total chunks in document: {len(all_chunks)}")

    for i, chunk in enumerate(all_chunks):
        if "29,915" in chunk or "29915" in chunk:
            print(f"R&D found at index {i}: {chunk[:300]}")
    later_chunks = all_chunks[30:65]   # financial statements start here
    all_financial = [c for c in later_chunks if is_financial_chunk(c)]
    print(f"Financial chunks found: {len(all_financial)}")
    documents = all_financial[:50]
    print("\nSample chunks indexed:")
    for i, doc in enumerate(documents[:5]):
        print(f"\nChunk {i}: {doc[:150]}")
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("multi_hop")
    for i, doc in enumerate(documents):
        collection.add(
            ids=[str(i)],
            embeddings=[get_embedding(doc)],
            documents=[doc],
            metadatas=[{"chunk_index": i}]
        )
    print(f"Indexed {len(documents)} chunks.")

    # Multi-hop questions
    queries = [
        "How did Apple's R&D spending compare to its net income in 2023?",
        "What was Apple's revenue and how did it change compared to 2022?",
    ]

    for query in queries:
        result = multi_hop_query(query, documents, collection)
        print(f"\n{'='*60}")
        print(f"FINAL ANSWER:\n{result['final_answer']}")
        print(f"\nSub-question results:")
        for r in result['sub_results']:
            print(f"  Q: {r['sub_question']}")
            print(f"  A: {r['answer']} (confidence: {r['confidence']})")