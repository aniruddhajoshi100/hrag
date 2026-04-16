import os
import textwrap
import time
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

load_dotenv()

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "academic_papers"
groq_api_key = os.getenv("GROQ_API_KEY", "")

if not groq_api_key:
    print("Please set GROQ_API_KEY in .env")
    exit(1)

# --- 1. DYNAMIC ROUTER SCHEMA ---
class SearchFilters(BaseModel):
    """Schema for extracting metadata filters from a user's natural language query."""
    target_title: str | None = Field(
        default=None, 
        description="Identify which specific paper is being asked about. MUST exactly match one of the titles provided in the system prompt. If no specific paper is mentioned, return None."
    )

# --- 2. VECTORSTORE & DYNAMIC TITLE EXTRACTION ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)

# Scan the metadata of all chunks to find unique titles
try:
    data = vectorstore.get(include=["metadatas"])
    unique_titles = list(set(meta.get("title") for meta in data["metadatas"] if meta and "title" in meta))
except Exception:
    unique_titles = []

if not unique_titles:
    print("⚠️ WARNING: No titles found in the database. Did you re-ingest your data with the new chunks.json?")

llm = ChatGroq(temperature=0.0, api_key=SecretStr(groq_api_key), model="llama-3.3-70b-versatile")

# --- 3. NOVEL RAG CHAIN BUILDER ---
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
    
    # Updated to use {title} instead of {source}
    document_prompt = PromptTemplate(
        input_variables=["page_content", "title", "section"],
        template="[PATH: {title} -> {section}]\n{page_content}\n"
    )
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt, document_prompt=document_prompt)
    return create_retrieval_chain(retriever, document_chain)

# --- 4. EXECUTION FUNCTIONS ---
def run_naive_rag(query: str):
    start = time.time()
    prompt = ChatPromptTemplate.from_template(
        """You are a precise academic research assistant. Use ONLY the following context from academic papers to answer the query. 
        If you cannot find the answer in the context, strictly output "Insufficient data to answer this query based on the retrieved context."
        
        Context:
        {context}
        
        Question: {input}
        
        Answer:"""
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    chain = create_retrieval_chain(retriever, document_chain)
    ans = chain.invoke({"input": query})
    return ans, time.time() - start

def run_novel_rag(query: str):
    start = time.time()
    
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
    
    chroma_filter = {}
    # Use exact title matching
    if extracted_filters.target_title and extracted_filters.target_title in unique_titles:
        chroma_filter["title"] = extracted_filters.target_title 

    search_kwargs: dict[str, Any] = {"k": 10} 
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter 
        
    test_retriever = vectorstore.as_retriever(search_kwargs={"k": 1, "filter": chroma_filter})
    if chroma_filter and len(test_retriever.invoke(query)) == 0:
        search_kwargs.pop("filter", None)
        chroma_filter = {} 
        
    chain = build_chain(llm, vectorstore, search_kwargs)
    response = chain.invoke({"input": query})
    
    return response, time.time() - start, chroma_filter

# --- 5. TEST QUERIES ---
test_queries = [
    {
        "question": "In the Attention paper, what BLEU score did the model achieve on the English-to-German translation task?",
        "truth": "It achieved a 28.4 BLEU score."
    },
    {
        "question": "According to the Chain of Thought paper, what is the most frequent error pattern in Commonsense Reasoning?",
        "truth": "Commonsense mistake (producing a flexible and reasonable chain of thought but reaching the wrong answer)."
    },
    {
        "question": "In the QLoRA paper, what are the three specific innovations introduced to save memory?",
        "truth": "4-bit NormalFloat (NF4), Double Quantization, and Paged Optimizers."
    },
    {
        "question": "According to the Attention paper, why do they scale the dot products by 1/sqrt(d_k)?",
        "truth": "To counteract the effect where large dot products push the softmax function into regions with extremely small gradients."
    },
    {
        "question": "In the Chain of Thought paper, what happens to performance on out-of-domain (OOD) test sets compared to standard prompting?",
        "truth": "CoT prompting achieves upward scaling curves, whereas standard prompting performs the worst and fails tasks."
    },
    {
        "question": "In the Attention paper, what is the exact dimensionality of the inner-layer in the position-wise feed-forward networks (d_ff)?",
        "truth": "The inner-layer has a dimensionality of d_ff = 2048."
    },
    {
        "question": "According to the QLoRA paper, what is the memory footprint of a deployed 7B Guanaco model?",
        "truth": "It requires just 5 GB of memory."
    },
    {
        "question": "In the Chain of Thought paper, what four specific datasets are used to evaluate Arithmetic Reasoning?",
        "truth": "GSM8K, SVAMP, ASDiv, and MAWPS."
    },
    {
        "question": "Based on the QLoRA paper, what specific dataset was used to train the Guanaco models, and what does it consist of?",
        "truth": "The OASST1 dataset, which is a multilingual collection of crowd-sourced multiturn dialogs."
    }
]

def _print_answer_block(title: str, answer: str, width: int = 110) -> None:
    print(f"{title}:")
    clean_answer = answer.strip()
    if not clean_answer:
        print("  [empty answer]")
        return
    for paragraph in clean_answer.splitlines():
        wrapped = textwrap.wrap(paragraph, width=width) or [""]
        for line in wrapped:
            print(f"  {line}")

def evaluate():
    print("Starting Comparison Evaluation...\n")
    print("=" * 80)
    
    for i, item in enumerate(test_queries):
        q = item["question"]
        t = item["truth"]
        print(f"Test {i+1}/{len(test_queries)}")
        print(f"Query: {q}")
        print(f"Expected Truth: {t}\n")
        
        # NAIVE
        res_naive, time_naive = run_naive_rag(q)
        ans_n = res_naive["answer"]
        _print_answer_block("Naive Answer", ans_n)
        print(f"  Time: {time_naive:.2f}s\n")
        
        # NOVEL
        res_novel, time_novel, filters = run_novel_rag(q)
        ans_nov = res_novel["answer"]
        _print_answer_block("Novel Answer", ans_nov)
        print(f"  Filter Extracted: {filters}")
        print(f"  Time: {time_novel:.2f}s\n")
        print("=" * 80)

if __name__ == "__main__":
    evaluate()