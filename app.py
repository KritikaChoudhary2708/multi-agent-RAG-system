import os 
import streamlit as st
import chromadb
from pypdf import PdfReader
from  sentence_transformers import SentenceTransformer
from ingestion_agent import fetch_sec_filing, chunk_text
from retrieval_agent import hybrid_search
from synthesis_agent import synthesize
from query_decomposition import multi_hop_query
from dotenv import load_dotenv

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Page config

st.set_page_config(
    page_title="Panchayat",
    page_icon="",
    layout="wide"
)

st.title("Panchayat")
st.caption("A multi agent RAG system: Upload -> Ask -> cite.  "
                            "100% grounded answers. No hallucinations.")

# Session state to persist data across interactions

if "documents" not in st.session_state:
          st.session_state.documents = []
if "collection" not in st.session_state:
          st.session_state.collection = None
if "doc_loaded" not in st.session_state:
          st.session_state.doc_loaded = False


# Helper : get embedding
def get_embedding(text: str)-> list[float]:
          return embedder.encode(text).tolist()

#extract text from pdf
def extract_pdf_text(pdf_file):
          reader = PdfReader(pdf_file)
          text = ""
          for page in reader.pages:
                    if page.extract_text():
                              text += page.extract_text() + "\n"
          return text

def ingest_document(text:str, source: str):
          chunks = chunk_text(text, chunk_size = 500, overlap = 50)

          #filter junk chunks
          clean_chunks=[
                    c for c in chunks
                    if len(c.split()) > 20 and "0000320193" not in c
          ]
          chroma_client = chromadb.Client()
          try:
                    chroma_client.delete_collection("document")
          except:
                    pass
          
          collection = chroma_client.create_collection("document")

          for i, chunk in enumerate(clean_chunks):
                    embedding = get_embedding(chunk)
                    collection.add(
                              documents=[chunk],
                              embeddings=[embedding],
                              ids=[f"{source}_{i}"],
                              metadatas=[{"source": source, "chunk": i}]
                    )
          return clean_chunks, collection

#sidebar

st.sidebar.header("load doc")

if st.sidebar.button("Clear / Reset All"):
          st.session_state.clear()
          st.rerun()

def handle_input_change():
          if "query_input" in st.session_state:
                    st.session_state.query_input = ""

input_type = st.sidebar.radio(
          "Choose input type",["URL", "Upload PDF"],
          on_change=handle_input_change
)

if input_type == "URL":
          url = st.sidebar.text_input("Paste document URL:")
          if st.sidebar.button("Load URL"):
                    with st.spinner("Fetching and processing document..."):
                              try:
                                        text = fetch_sec_filing(url)
                                        documents, collection = ingest_document(text, url)

                                        st.session_state.documents = documents
                                        st.session_state.collection = collection
                                        st.session_state.doc_loaded = True
                                        st.success("Document loaded successfully!")
                              except Exception as e:
                                        st.error(f"Error loading document: {e}")
elif input_type == "Upload PDF":
          pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
          load_btn = st.sidebar.button("Load PDF")   # ← show button always

          if pdf_file and load_btn:
                    with st.spinner("Processing PDF..."):
                              try:
                                        text = extract_pdf_text(pdf_file)
                                        documents, collection = ingest_document(text, source=pdf_file.name)
                                        st.session_state.documents = documents
                                        st.session_state.collection = collection
                                        st.session_state.doc_loaded = True
                                        st.sidebar.success("PDF loaded successfully!")
                              except Exception as e:
                                        st.sidebar.error(f"Error loading pdf: {e}")

if st.session_state.doc_loaded:
          st.success(f"Document loaded-- {len(st.session_state.documents)} chunks indexed")
          #Query mode
          mode = st.radio(
                    "Query mode:", ["Simple", "Multi-hop"]
          )

          query=st.text_input("Ask a question", key="query_input")
          if st.button("Ask") and query:
                    with st.spinner("Retrieving & synthesize..."):
                              if mode == "Simple":
                                        top_chunks = hybrid_search(
                                                  query,
                                                  st.session_state.documents,
                                                  st.session_state.collection,
                                                  top_k=5
                                        )
                                        result = synthesize(query, top_chunks)

                                        st.subheader("Answer")
                                        st.write(result["answer"])

                                        col1, col2 = st.columns(2)
                                        col1.metric("Confidence", f"{result['confidence']:.0%}")
                                        col2.metric("Chunks Used", result["chunks_used"])

                                        with st.expander("View retrieved chunks"):
                                                  for i, chunk in enumerate(top_chunks):
                                                            st.markdown(f"**Chunk {i+1}:**")
                                                            st.write(chunk[:300] + "...")

                              else:
                                        result = multi_hop_query(
                                                  query,
                                                  st.session_state.documents,
                                                  st.session_state.collection
                                        )

                                        st.subheader("Final Answer")
                                        st.write(result["final_answer"])

                                        with st.expander("View sub-question breakdown"):
                                                  for r in result["sub_results"]:
                                                            st.markdown(f"**Q:** {r['sub_question']}")
                                                            st.markdown(f"**A:** {r['answer']}")
                                                            st.markdown(f"Confidence: {r['confidence']:.0%}")
                                                            st.divider()

else:
          st.info("👈 Load a document from the sidebar to get started")

          # Show example queries
          st.subheader("Example queries you can try:")
          st.markdown("""
          - *What was Apple's total revenue in 2023?*
          - *How much did Apple spend on R&D?*
          - *How did Apple's revenue compare to 2022?*
          - *What were Apple's main business risks?*
          """)
          
          

          