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
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

load_dotenv()

st.set_page_config(page_title="Academic RAG Explorer", layout="wide")

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "academic_papers"

env_api_key = os.getenv("GROQ_API_KEY", "")

# --- 1. DYNAMIC ROUTER SCHEMA ---
class SearchFilters(BaseModel):
    """Schema for extracting metadata filters from a user's natural language query."""
    target_title: str | None = Field(
        default=None, 
        description="Identify which specific paper is being asked about. MUST exactly match one of the titles provided in the system prompt. If no specific paper is mentioned, return None."
    )

# --- Sidebar Configuration ---
st.sidebar.title("RAG Settings")
if env_api_key:
    groq_api_key = env_api_key
    st.sidebar.success("API Key loaded from environment.")
else:
    groq_api_key = st.sidebar.text_input("Groq API Key (Not found in .env)", type="password", placeholder="gsk_...")

model_name = st.sidebar.selectbox("LLM Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"])

# --- Core RAG Setup Functions ---
@st.cache_resource
def load_vectorstore_and_titles():
    """Loads the vectorstore and dynamically extracts all unique paper titles from the database."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(CHROMA_DB_DIR):
        return None, []
    db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)
    
    # Scan the metadata of all chunks to find unique titles
    try:
        data = db.get(include=["metadatas"])
        unique_titles = list(set(meta.get("title") for meta in data["metadatas"] if meta and "title" in meta))
    except Exception:
        unique_titles = []
        
    return db, unique_titles

def build_chain(llm, vectorstore, search_kwargs):
    prompt = ChatPromptTemplate.from_template(
        """You are a highly precise academic research assistant. Use ONLY the following context from academic papers to answer the query. 
        Each piece of context includes its hierarchical path (Title -> Section). Pay close attention to this path to understand where the information comes from.
        
        If you cannot find the answer in the context, strictly output "Insufficient data to answer this query based on the retrieved context." Do not hallucinate.
        
        Context:
        {context}
        
        Question: {input}
        
        Answer:"""
    )
    
    # Notice we updated the input variables to use your new 'title' metadata field!
    document_prompt = PromptTemplate(
        input_variables=["page_content", "title", "section"],
        template="[PATH: {title} -> {section}]\n{page_content}\n"
    )
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    
    document_chain = create_stuff_documents_chain(
        llm=llm, 
        prompt=prompt,
        document_prompt=document_prompt 
    )
    
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

# --- Main App Logic ---
st.title("Hierarchical RAG for Academic Documents")
st.markdown("Search across papers. The AI will automatically route using metadata titles instead of generic filenames.")

# Fetch both the database AND the list of valid titles
vectorstore, unique_titles = load_vectorstore_and_titles()

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
                
                # 2. The Intelligent Router Step
                # Dynamically inject the exact database titles into the Router's instructions
                valid_titles_str = "\n".join([f"- {t}" for t in unique_titles])
                router_system_prompt = f"""You are an intelligent routing agent. Your job is to extract the target paper title from the user's query.
                You MUST select EXACTLY from this list of valid paper titles (or return None if the query is general):
                {valid_titles_str}
                """
                
                router_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", router_system_prompt),
                        ("user", "{query}"),
                    ]
                )
                
                structured_llm = router_prompt | llm.with_structured_output(SearchFilters)
                extracted_filters = structured_llm.invoke({"query": query})
                
                # --- 3. DYNAMIC METADATA FILTER LOGIC ---
                chroma_filter = {}

                # If the AI successfully picked a known title from the DB
                if extracted_filters.target_title and extracted_filters.target_title in unique_titles:
                    chroma_filter["title"] = extracted_filters.target_title 

                # Configure initial search kwargs
                search_kwargs: dict[str, Any] = {"k": 10} 
                if chroma_filter:
                    search_kwargs["filter"] = chroma_filter 
                    st.success(f"🤖 AI Auto-Filtered by exact title: {chroma_filter}")
                else:
                    st.info("🤖 Executing global vector search.")
                    
                # --- THE SAFETY NET ---
                test_retriever = vectorstore.as_retriever(search_kwargs={"k": 1, "filter": chroma_filter})
                if chroma_filter and len(test_retriever.invoke(query)) == 0:
                    st.warning(f"⚠️ Could not find exact match for {chroma_filter} in database. Falling back to global search...")
                    search_kwargs.pop("filter", None)
                # ----------------------------------------
                
                st.spinner("Executing search and generating answer...")
                
                # 4. Create the Chain and generate
                chain = build_chain(llm, vectorstore, search_kwargs)
                response = chain.invoke({"input": query})
                
                # 5. Display Results
                st.markdown("### Answer")
                st.write(response["answer"])
                
                st.markdown("### Source Evidence Retrieved")
                for i, doc in enumerate(response["context"]):
                    # Show the actual title instead of 'p1.pdf' in the UI expander
                    with st.expander(f"Source {i+1}: {doc.metadata.get('title', 'Unknown Title')} | Section: {doc.metadata.get('section', 'Main Text')}"):
                        st.json(doc.metadata)  
                        st.write(doc.page_content) 
                        
            except Exception as e:
                st.error(f"An error occurred during query execution: {e}")