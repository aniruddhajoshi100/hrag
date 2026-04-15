import os
from typing import Any
from dotenv import load_dotenv
import streamlit as st
from pydantic import BaseModel, Field, SecretStr
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

# --- 1. Define the AI Router Schema ---
class SearchFilters(BaseModel):
    """Schema for extracting metadata filters from a user's natural language query."""
    target_source: str | None = Field(
        default=None, 
        description="The specific paper, filename, or author mentioned (e.g., 'p1.pdf', 'Attention paper'). If no specific paper is mentioned, return None."
    )
    target_section: str | None = Field(
        default=None, 
        description="The specific section of the paper mentioned (e.g., 'Methodology', 'Abstract', 'Results', 'Conclusion'). If no section is mentioned, return None."
    )

# --- Sidebar Configuration ---
st.sidebar.title("RAG Settings")
if env_api_key:
    groq_api_key = env_api_key
    st.sidebar.success("API Key loaded from environment.")
else:
    groq_api_key = st.sidebar.text_input("Groq API Key (Not found in .env)", type="password", placeholder="gsk_...")

model_name = st.sidebar.selectbox("LLM Model", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"])

# Notice: We completely removed the manual Target Section/Source text inputs from the sidebar!

# --- Core RAG Setup Functions ---
@st.cache_resource
def load_vectorstore():
    # Use the same free local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(CHROMA_DB_DIR):
        return None
    db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)
    return db

def build_chain(llm, vectorstore, search_kwargs):
    # Combine user query explicitly to the LLM alongside retrieved context
    prompt = ChatPromptTemplate.from_template(
        """You are a highly precise academic research assistant. Use ONLY the following context from academic papers to answer the query. 
        If you cannot find the answer in the context, strictly output "Insufficient data to answer this query based on the retrieved context." Do not hallucinate external knowledge.
        
        Context:
        {context}
        
        Question: {input}
        
        Answer:"""
    )
    
    # Using MMR to diversify results
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# --- Main App Logic ---
st.title("Hierarchical RAG for Academic Documents")
st.markdown("Search across papers. The AI will automatically detect if you are asking about a specific section or paper.")

vectorstore = load_vectorstore()

if not vectorstore:
    st.warning("No Vector Database found. Please make sure you have run the ingestion script first to generate `./chroma_db`.")
    st.stop()

query = st.text_input("Ask a technical question about the papers:")

if st.button("Search & Generate") and query:
    if not groq_api_key.strip():
        st.error("Please enter your Groq API Key in the sidebar.")
    elif not vectorstore:
        st.error("Vector database is unavailable.")
    else:
        with st.spinner("Analyzing query and extracting routing filters..."):
            try:
                # 1. Initialize Groq Client
                llm = ChatGroq(temperature=0.0, api_key=SecretStr(groq_api_key), model=model_name)
                
                # --- NEW: The Intelligent Router Step ---
                # Force the LLM to output a JSON object matching our SearchFilters schema
                structured_llm = llm.with_structured_output(SearchFilters)
                extracted_filters = structured_llm.invoke(query)
                
                # Build the Chroma dictionary dynamically based on what the AI found
                chroma_filter = {}
                if extracted_filters.target_source:
                    chroma_filter["source"] = {"$contains": extracted_filters.target_source}
                if extracted_filters.target_section:
                    chroma_filter["section"] = {"$contains": extracted_filters.target_section}
                
                # Display to the user what the AI decided to filter by
                if chroma_filter:
                    st.success(f"AI Auto-Filtered by: {chroma_filter}")
                else:
                    st.info("AI detected no specific filters. Executing global search.")
                # ----------------------------------------
                
                # 2. Configure kwargs for LangChain's Chroma Retriever
                search_kwargs: dict[str, Any] = {"k": 4} 
                if chroma_filter:
                    search_kwargs["filter"] = chroma_filter 
                
                # 3. Create the Chain and generate
                chain = build_chain(llm, vectorstore, search_kwargs)
                response = chain.invoke({"input": query})
                
                # 4. Display Results
                st.markdown("### Answer")
                st.write(response["answer"])
                
                st.markdown("### Source Evidence Retrieved")
                for i, doc in enumerate(response["context"]):
                    with st.expander(f"Source {i+1}: {doc.metadata.get('source', 'Unknown File')} | Section: {doc.metadata.get('section', 'Main Text')}"):
                        st.json(doc.metadata)  
                        st.write(doc.page_content) 
                        
            except Exception as e:
                st.error(f"An error occurred during query execution: {e}")