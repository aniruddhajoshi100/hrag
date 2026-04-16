import os
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from app import SearchFilters, PaperKey, PAPER_ALIASES  # Importing from your app.py

load_dotenv()

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "academic_papers"
groq_api_key = os.getenv("GROQ_API_KEY", "")

if not groq_api_key:
    print("Please set GROQ_API_KEY in .env")
    exit(1)

# Initialize models
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)

# We use the versatile model for generating answers
llm = ChatGroq(temperature=0.0, api_key=SecretStr(groq_api_key), model="llama-3.3-70b-versatile")
# Fast model for routing
router_llm = ChatGroq(temperature=0.0, api_key=SecretStr(groq_api_key), model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """You are a precise academic research assistant. Use ONLY the following context from academic papers to answer the query. 
    If you cannot find the answer in the context, strictly output "Insufficient data to answer this query based on the retrieved context."
    
    Context:
    {context}
    
    Question: {input}
    
    Answer:"""
)
document_chain = create_stuff_documents_chain(llm, prompt)

# --- 1. NAIVE RAG (Baseline) ---
def run_naive_rag(query: str):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    chain = create_retrieval_chain(retriever, document_chain)
    return chain.invoke({"input": query})

# --- 2. MMR RAG ---
def run_mmr_rag(query: str):
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    chain = create_retrieval_chain(retriever, document_chain)
    return chain.invoke({"input": query})

# --- 3. NOVEL RAG (Hierarchical + Routing) ---
def run_novel_rag(query: str):
    structured_llm = router_llm.with_structured_output(SearchFilters)
    extracted_filters = structured_llm.invoke(query)
    
    chroma_filter = {}
    if extracted_filters.target_source != PaperKey.unknown:
        mapped_source = PAPER_ALIASES.get(extracted_filters.target_source)
        if mapped_source:
            chroma_filter["source"] = mapped_source 
            
    search_kwargs = {"k": 5}
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter
        
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    test_docs = retriever.invoke(query)
    
    # Fallback
    if chroma_filter and len(test_docs) == 0:
        search_kwargs.pop("filter", None)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
        
    chain = create_retrieval_chain(retriever, document_chain)
    return chain.invoke({"input": query})

# --- LLM-AS-A-JUDGE EVALUATION ---
class EvaluationScore(BaseModel):
    score: int = Field(description="Score from 1 to 10 evaluating the accuracy and relevance of the answer.")
    reasoning: str = Field(description="Brief explanation for the score.")

eval_prompt = ChatPromptTemplate.from_template(
    """You are an expert evaluator. Evaluate the AI's answer to the user's question based on the expected ground truth.
    Did the AI successfully extract the right information? Did it hallucinate? Give a score from 1 to 10.
    
    Question: {question}
    Expected Truth: {ground_truth}
    AI Answer: {answer}
    """
)
eval_chain = eval_prompt | llm.with_structured_output(EvaluationScore)

# --- TEST DATASET ---
test_queries = [
    {
        "question": "In the QLoRA paper, what is the exact memory footprint of a 65B parameter model?",
        "truth": "The 65B parameter model requires a 48GB GPU for finetuning."
    },
    {
        "question": "What is the core mechanism introduced in the Attention paper?",
        "truth": "Self-attention mechanism computing scaled dot-product attention without recurrence or convolutions."
    }
]

def evaluate():
    print("Starting Comparison Evaluation...\n")
    for item in test_queries:
        q = item["question"]
        t = item["truth"]
        print(f"--- Query: {q} ---")
        print(f"Expected: {t}\n")
        
        for name, func in [("Naive RAG", run_naive_rag), ("MMR RAG", run_mmr_rag), ("Novel Hierarchical RAG", run_novel_rag)]:
            print(f"Running {name}...")
            start = time.time()
            try:
                res = func(q)
                ans = res["answer"]
                ev = eval_chain.invoke({"question": q, "ground_truth": t, "answer": ans})
                
                print(f"  Score: {ev.score}/10")
                print(f"  Time: {time.time() - start:.2f}s")
                print(f"  Reason: {ev.reasoning}\n")
            except Exception as e:
                print(f"  Failed: {e}\n")

if __name__ == "__main__":
    evaluate()
