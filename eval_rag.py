import os
import textwrap
import time
from dataclasses import dataclass
from typing import Any, cast
from enum import Enum
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

# --- 1. STRICT MULTIPLE CHOICE & ALIASES (Exact match to app.py) ---
class PaperKey(str, Enum):
    qlora = "qlora"
    attention = "attention"
    chain_of_thought = "chain_of_thought"
    unknown = "unknown"

PAPER_ALIASES = {
    PaperKey.attention: "p1.pdf",
    PaperKey.qlora: "p2.pdf",
    PaperKey.chain_of_thought: "p3.pdf"
}

class SearchFilters(BaseModel):
    """Schema for extracting metadata filters from a user's natural language query."""
    target_source: PaperKey = Field(
        default=PaperKey.unknown, 
        description="Identify which specific paper is being asked about. Select from the provided enum list. If no specific paper is mentioned, select 'unknown'."
    )
    target_section: str | None = Field(
        default=None, 
        description="The specific section of the paper mentioned. If no section is mentioned, return None."
    )

# --- 2. VECTORSTORE & LLM SETUP ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)

llm = ChatGroq(temperature=0.0, api_key=SecretStr(groq_api_key), model="llama-3.3-70b-versatile")

# --- 3. NOVEL RAG CHAIN BUILDER (Exact match to app.py) ---
def build_chain(llm, vectorstore, search_kwargs):
    prompt = ChatPromptTemplate.from_template(
        """You are a highly precise academic research assistant. Use ONLY the following context from academic papers to answer the query. 
        Each piece of context includes its hierarchical path (Source File -> Section). 
        
        CRITICAL FILE MAPPING:
        The user will often refer to papers by their common names. Use this exact mapping to associate the Source File paths with the user's question:
        - "p1.pdf" IS the "Attention Is All You Need" / "Attention" paper.
        - "p2.pdf" IS the "QLoRA" paper.
        - "p3.pdf" IS the "Chain of Thought" (CoT) paper.
        
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
    
    return create_retrieval_chain(retriever, document_chain)

# --- 4. EXECUTION FUNCTIONS ---
def run_naive_rag(query: str):
    start = time.time()
    # Basic prompt with no path guidance and no file mapping
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
    
    # 1. Intelligent Router Step
    structured_llm = llm.with_structured_output(SearchFilters)
    extracted_filters = structured_llm.invoke(query)
    
    # 2. Enum Filter Logic
    chroma_filter = {}
    if extracted_filters.target_source != PaperKey.unknown:
        mapped_source = PAPER_ALIASES.get(extracted_filters.target_source)
        if mapped_source:
            chroma_filter["source"] = mapped_source 

    search_kwargs: dict[str, Any] = {"k": 10} 
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter 
        
    # 3. The Safety Net
    test_retriever = vectorstore.as_retriever(search_kwargs={"k": 1, "filter": chroma_filter})
    if chroma_filter and len(test_retriever.invoke(query)) == 0:
        search_kwargs.pop("filter", None)
        chroma_filter = {} # Clear to indicate global search
        
    # 4. Chain Creation and Generation
    chain = build_chain(llm, vectorstore, search_kwargs)
    response = chain.invoke({"input": query})
    
    return response, time.time() - start, chroma_filter

# --- 5. EVALUATION LOGIC ---
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
        "question": "In the QLoRA paper, what is the memory footprint of a 65B parameter model?",
        "truth": "The 65B parameter model requires <48GB GPU memory for finetuning."
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

@dataclass
class EvalResult:
    question: str
    naive_answer: str
    novel_answer: str
    naive_score: int
    naive_time: float
    novel_score: int
    novel_time: float
    filter_used: str

def _truncate(text: str, max_len: int = 58) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."

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

def _print_score_table(results: list[EvalResult]) -> None:
    headers = [
        "Query",
        "Naive Score",
        "Naive Time(s)",
        "Novel Score",
        "Novel Time(s)",
        "Filter",
    ]

    rows: list[list[str]] = []
    for result in results:
        rows.append(
            [
                _truncate(result.question),
                f"{result.naive_score}/10",
                f"{result.naive_time:.2f}",
                f"{result.novel_score}/10",
                f"{result.novel_time:.2f}",
                result.filter_used,
            ]
        )

    avg_naive_score = sum(r.naive_score for r in results) / len(results)
    avg_novel_score = sum(r.novel_score for r in results) / len(results)
    avg_naive_time = sum(r.naive_time for r in results) / len(results)
    avg_novel_time = sum(r.novel_time for r in results) / len(results)

    avg_row = [
        "AVERAGE",
        f"{avg_naive_score:.1f}/10",
        f"{avg_naive_time:.2f}",
        f"{avg_novel_score:.1f}/10",
        f"{avg_novel_time:.2f}",
        "-",
    ]

    widths = [
        max(
            len(headers[i]),
            max((len(row[i]) for row in rows), default=0),
            len(avg_row[i]),
        )
        for i in range(len(headers))
    ]

    def make_separator(fill: str = "-") -> str:
        return "+" + "+".join(fill * (w + 2) for w in widths) + "+"

    def make_row(values: list[str]) -> str:
        padded = [f" {values[i]:<{widths[i]}} " for i in range(len(values))]
        return "|" + "|".join(padded) + "|"

    print("\nFINAL EVALUATION SCORES")
    print(make_separator("="))
    print(make_row(headers))
    print(make_separator("-"))
    for row in rows:
        print(make_row(row))
    print(make_separator("-"))
    print(make_row(avg_row))
    print(make_separator("="))

def evaluate():
    print("Starting Comparison Evaluation...\n")
    
    results: list[EvalResult] = []
    
    for item in test_queries:
        q = item["question"]
        t = item["truth"]
        print(f"--- Query: {q} ---")
        
        # NAIVE
        print(f"Running Naive RAG...")
        res_naive, time_naive = run_naive_rag(q)
        ans_n = res_naive["answer"]
        ev_naive = cast(EvaluationScore, eval_chain.invoke({"question": q, "ground_truth": t, "answer": ans_n}))
        _print_answer_block("Naive Answer", ans_n)
        print(f"  Score: {ev_naive.score}/10 | Time: {time_naive:.2f}s")
        
        # NOVEL
        print(f"Running Novel Hierarchical RAG...")
        res_novel, time_novel, filters = run_novel_rag(q)
        ans_nov = res_novel["answer"]
        ev_novel = cast(EvaluationScore, eval_chain.invoke({"question": q, "ground_truth": t, "answer": ans_nov}))
        _print_answer_block("Novel Answer", ans_nov)
        print(f"  Filter Extracted: {filters}")
        print(f"  Score: {ev_novel.score}/10 | Time: {time_novel:.2f}s")

        results.append(
            EvalResult(
                question=q,
                naive_answer=ans_n,
                novel_answer=ans_nov,
                naive_score=ev_naive.score,
                naive_time=time_naive,
                novel_score=ev_novel.score,
                novel_time=time_novel,
                filter_used=str(filters.get("source", "None")),
            )
        )
        print()
    _print_score_table(results)
    print()

if __name__ == "__main__":
    evaluate()