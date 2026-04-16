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
from app import SearchFilters, PaperKey, PAPER_ALIASES

load_dotenv()

CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "academic_papers"
groq_api_key = os.getenv("GROQ_API_KEY", "")

if not groq_api_key:
    print("Please set GROQ_API_KEY in .env")
    exit(1)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)

llm = ChatGroq(temperature=0.0, api_key=SecretStr(groq_api_key), model="llama-3.3-70b-versatile")
router_llm = ChatGroq(temperature=0.0, api_key=SecretStr(groq_api_key), model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template(
    """You are a precise academic research assistant. Use ONLY the following context from academic papers to answer the query. 
    If you cannot find the answer in the context, strictly output "Insufficient data to answer this query based on the retrieved context."
    
    Context:
    {context}
    
    Question: {input}
    
    Answer:"""
)
document_chain = create_stuff_documents_chain(llm, prompt)

def run_naive_rag(query: str):
    start = time.time()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    chain = create_retrieval_chain(retriever, document_chain)
    ans = chain.invoke({"input": query})
    return ans, time.time() - start

def run_novel_rag(query: str):
    start = time.time()
    structured_llm = router_llm.with_structured_output(SearchFilters)
    extracted_filters = structured_llm.invoke(query)
    
    chroma_filter = {}
    if extracted_filters.target_source != PaperKey.unknown:
        mapped_source = PAPER_ALIASES.get(extracted_filters.target_source)
        if mapped_source:
            chroma_filter["source"] = mapped_source 
            
    search_kwargs = {"k": 10, "fetch_k": 30}
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter
        
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    test_docs = retriever.invoke(query)
    
    if chroma_filter and len(test_docs) == 0:
        search_kwargs.pop("filter", None)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
        
    chain = create_retrieval_chain(retriever, document_chain)
    ans = chain.invoke({"input": query})
    return ans, time.time() - start, chroma_filter

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

test_queries = [
    {
        "question": "In the QLoRA paper, what is the exact memory footprint of a 65B parameter model?",
        "truth": "The 65B parameter model requires a 48GB GPU for finetuning."
    },
    {
        "question": "What is the core mechanism introduced in the Attention paper?",
        "truth": "Self-attention mechanism computing scaled dot-product attention without recurrence or convolutions."
    },
    {
        "question": "According to the Chain of Thought paper, how does it improve reasoning?",
        "truth": "It prompts the model to generate intermediate reasoning steps before giving the final answer."
    },
    {
        "question": "Summarize the primary advantage of QLoRA over standard fine-tuning.",
        "truth": "QLoRA reduces memory usage to enable fine-tuning large models on a single GPU without losing performance."
    },
    {
        "question": "What scaling problem does QLoRA address?",
        "truth": "It addresses the memory constraints that prevent training large parameter models on standard hardware."
    }
]

def evaluate():
    print("Starting Comparison Evaluation...\n")
    
    results = []
    
    for item in test_queries:
        q = item["question"]
        t = item["truth"]
        print(f"--- Query: {q} ---")
        
        row = {"Query": q[:35] + "..."}
        
        # NAIVE
        print(f"Running Naive RAG...")
        res_naive, time_naive = run_naive_rag(q)
        ans_n = res_naive["answer"]
        ev_naive = eval_chain.invoke({"question": q, "ground_truth": t, "answer": ans_n})
        row["Naive_Score"] = ev_naive.score
        row["Naive_Time"] = time_naive
        print(f"  Score: {ev_naive.score}/10 | Time: {time_naive:.2f}s")
        
        # NOVEL
        print(f"Running Novel Hierarchical RAG...")
        res_novel, time_novel, filters = run_novel_rag(q)
        ans_nov = res_novel["answer"]
        ev_novel = eval_chain.invoke({"question": q, "ground_truth": t, "answer": ans_nov})
        row["Novel_Score"] = ev_novel.score
        row["Novel_Time"] = time_novel
        row["Filter_Used"] = str(filters.get("source", "None"))
        print(f"  Filter Extracted: {filters}")
        print(f"  Score: {ev_novel.score}/10 | Time: {time_novel:.2f}s")
            
        results.append(row)
        print()

    # Create the final tabular result
    print("\n" + "="*95)
    print(f"{'FINAL EVALUATION RESULTS':^95}")
    print("="*95)
    
    # Print header
    header = f"| {'Query Prefix':<38} | {'Naive (Score / Time)':<20} | {'Novel (Score / Time)':<20} | {'Filter Hit':<10} |"
    print(header)
    print("-" * len(header))
    
    total_naive_s = 0
    total_novel_s = 0
    total_naive_t = 0
    total_novel_t = 0
    
    for res in results:
        n_str = f"{res['Naive_Score']}/10 ({res['Naive_Time']:.1f}s)"
        nov_str = f"{res['Novel_Score']}/10 ({res['Novel_Time']:.1f}s)"
        print(f"| {res['Query']:<38} | {n_str:<20} | {nov_str:<20} | {res['Filter_Used'][:10]:<10} |")
        total_naive_s += res['Naive_Score']
        total_novel_s += res['Novel_Score']
        total_naive_t += res['Naive_Time']
        total_novel_t += res['Novel_Time']
        
    print("-" * len(header))
    
    avg_n_str = f"{total_naive_s/len(results):.1f}/10 ({total_naive_t/len(results):.1f}s)"
    avg_nov_str = f"{total_novel_s/len(results):.1f}/10 ({total_novel_t/len(results):.1f}s)"
    print(f"| {'AVERAGE':<38} | {avg_n_str:<20} | {avg_nov_str:<20} | {'-':<10} |")
    print("="*95 + "\n")

if __name__ == "__main__":
    evaluate()
