import os
from typing import Any
from dotenv import load_dotenv
import streamlit as st
from pydantic import SecretStr
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()


st.set_page_config(page_title="Academic RAG Explorer", layout="wide")

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "academic_papers"

env_api_key = os.getenv("GROQ_API_KEY", "")

# --- Sidebar Configuration ---
st.sidebar.title("RAG Settings")
if env_api_key:
    groq_api_key = env_api_key
    st.sidebar.success("API Key loaded from environment.")
else:
    groq_api_key = st.sidebar.text_input("Groq API Key (Not found in .env)", type="password", placeholder="gsk_...")

model_name = st.sidebar.selectbox("LLM Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"])

# Sidebar Metadata Filters
st.sidebar.markdown("### Metadata Filters")
author_filter = st.sidebar.selectbox("Author", ["Any", "Alice Scholar", "Bob Researcher", "Charlie Academic"])
year_filter = st.sidebar.selectbox("Year", ["Any", 2022, 2023, 2024])
topic_filter = st.sidebar.selectbox("Topic", ["Any", "Machine Learning", "Natural Language Processing", "Computer Vision"])

# --- Core RAG Setup Functions ---
@st.cache_resource
def load_vectorstore():
    # Use the same free local embedding model
    st.write("Loading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(CHROMA_DB_DIR):
        return None
    db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)
    return db

def build_chain(llm, vectorstore, search_kwargs):
    # Combine user query explicitly to the LLM alongside retrieved context
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful academic research assistant. Use the following context from academic papers to answer the query accurately. 
        If you cannot find the answer in the context, say "I don't know based on the provided papers."
        
        Context:
        {context}
        
        Question: {input}
        
        Answer:"""
    )
    
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# --- Main App Logic ---
st.title("📚 Hierarchical RAG for Academic Documents")
st.markdown("Search across papers and filter specifically by their section context, author, year, or topic.")

vectorstore = load_vectorstore()

if not vectorstore:
    st.warning("No Vector Database found. Please make sure you have run `embed_index.py` first to generate `./chroma_db`.")
    st.stop()

# Build the filter dict dynamically from sidebar selections
chroma_filter = {}
if author_filter != "Any":
    chroma_filter["author"] = author_filter
if year_filter != "Any":
    chroma_filter["year"] = int(year_filter)
if topic_filter != "Any":
    chroma_filter["topic"] = topic_filter

query = st.text_input("Ask a question about the papers:")

if st.button("Search & Generate") and query:
    if not groq_api_key.strip():
        st.error("Please enter your Groq API Key in the sidebar.")
    elif not vectorstore:
        st.error("Vector database is unavailable.")
    else:
        with st.spinner("Searching and generating answer..."):
            try:
                # 1. Initialize Groq Client
                llm = ChatGroq(temperature=0.2, api_key=SecretStr(groq_api_key), model=model_name)
                
                # 2. Configure kwargs for LangChain's Chroma Retriever
                search_kwargs: dict[str, Any] = {"k": 3}
                if chroma_filter:
                    search_kwargs["filter"] = chroma_filter
                    st.info(f"Filtering context by: {chroma_filter}")
                else:
                    st.info("Searching all papers without metadata filters.")
                
                # 3. Create the Chain and generate
                chain = build_chain(llm, vectorstore, search_kwargs)
                response = chain.invoke({"input": query})
                
                # 4. Display Results
                st.markdown("### Answer")
                st.write(response["answer"])
                
                st.markdown("### Source Context Retrieved")
                for i, doc in enumerate(response["context"]):
                    with st.expander(f"Source {i+1}: {doc.metadata.get('source', 'Unknown')} - {doc.metadata.get('section', 'Main')}"):
                        st.json(doc.metadata)  # Show the metadata (Year, Author, Topic)
                        st.write(doc.page_content[:500] + "...") # Show snippet of the text
                        
            except Exception as e:
                st.error(f"An error occurred during query execution: {e}")
