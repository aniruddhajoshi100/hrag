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
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from rag_schema import PaperKey, PAPER_ALIASES, SearchFilters, ROUTER_SYSTEM_PROMPT

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

model_name = st.sidebar.selectbox("LLM Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"])

# --- Core RAG Setup Functions ---
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(CHROMA_DB_DIR):
        return None
    db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)
    return db

def build_chain(llm, vectorstore, search_kwargs):
    prompt = ChatPromptTemplate.from_template(
        """You are a highly precise academic research assistant. Use ONLY the following context from academic papers to answer the query. 
        Each piece of context includes its hierarchical path (Source File -> Section). Pay close attention to this path to understand where the information comes from.
        
        If you cannot find the answer in the context, strictly output "Insufficient data to answer this query based on the retrieved context." Do not hallucinate.
        
        Context:
        {context}
        
        Question: {input}
        
        Answer:"""
    )
    
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source", "section"],
        template="[PATH: {source} -> {section}]\n{page_content}\n"
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
                
                # 2. The Intelligent Router Step
                router_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", ROUTER_SYSTEM_PROMPT),
                        ("user", "{query}"),
                    ]
                )
                structured_llm = router_prompt | llm.with_structured_output(SearchFilters)
                extracted_filters = structured_llm.invoke({"query": query})
                
                # --- 3. THE ENUM FILTER LOGIC ---
                chroma_filter = {}

                # If the AI successfully picked a known paper...
                if extracted_filters.target_source != PaperKey.unknown:
                    # Get the exact filename from our alias dictionary
                    mapped_source = PAPER_ALIASES.get(extracted_filters.target_source)
                    
                    if mapped_source:
                        chroma_filter["source"] = mapped_source 

                # Configure initial search kwargs
                search_kwargs: dict[str, Any] = {"k": 10} # MMR Needs fetch_k to be higher!
                if chroma_filter:
                    search_kwargs["filter"] = chroma_filter 
                    st.success(f"🤖 AI Auto-Filtered by exact source: {chroma_filter}")
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
                    # Reverted to generic 'Unknown File' to avoid system identifiers
                    with st.expander(f"Source {i+1}: {doc.metadata.get('source', 'Unknown File')} | Section: {doc.metadata.get('section', 'Main Text')}"):
                        st.json(doc.metadata)  
                        st.write(doc.page_content) 
                        
            except Exception as e:
                st.error(f"An error occurred during query execution: {e}")