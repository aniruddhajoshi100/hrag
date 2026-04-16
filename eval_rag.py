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

# --- 1. STRICT MULTIPLE CHOICE & ALIASES ---
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
    target_source: PaperKey = Field(
        default=PaperKey.unknown, 
        description="Identify which specific paper is being asked about. Select from the provided enum list. If no specific paper is mentioned, select 'unknown'."
    )
    target_section: str | None = Field(default=None)

# --- 2. VECTORSTORE & LLM SETUP ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name=COLLECTION_NAME)

llm = ChatGroq(temperature=0.0, api_key=SecretStr(groq_api_key), model="llama-3.3-70b-versatile")

# --- 3. NOVEL RAG CHAIN BUILDER ---
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
    
    structured_llm = llm.with_structured_output(SearchFilters)
    extracted_filters = structured_llm.invoke(query)
    
    chroma_filter = {}
    if extracted_filters.target_source != PaperKey.unknown:
        mapped_source = PAPER_ALIASES.get(extracted_filters.target_source)
        if mapped_source:
            chroma_filter["source"] = mapped_source 

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

# --- 5. EVALUATION LOGIC (Pass/Fail) ---
class EvaluationScore(BaseModel):
    success: bool = Field(description="True if the AI successfully extracted the right information and gave a correct answer. False if it output 'Insufficient data', hallucinated, or gave a wrong answer.")
    reasoning: str = Field(description="Brief explanation for the Pass/Fail judgment.")

eval_prompt = ChatPromptTemplate.from_template(
    """You are an expert evaluator. Evaluate the AI's answer to the user's question based on the expected ground truth.
    Did the AI successfully extract the right information? Did it hallucinate? Output True for a Pass, False for a Fail.
    
    Question: {question}
    Expected Truth: {ground_truth}
    AI Answer: {answer}
    """
)
eval_chain = eval_prompt | llm.with_structured_output(EvaluationScore)

# ADVERSARIAL QUERIES: Designed to confuse Naive RAG by burying the context across multiple papers.
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
    }
]

@dataclass
class EvalResult:
    question: str
    naive_answer: str
    novel_answer: str
    naive_pass: bool
    naive_time: float
    novel_pass: bool
    novel_time: float
    filter_used: str

def _truncate(text: str, max_len: int = 56) -> str:
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
    headers = ["Query", "Naive Result", "Naive Time(s)", "Novel Result", "Novel Time(s)", "Filter Extracted"]

    rows: list[list[str]] = []
    for result in results:
        rows.append([
            _truncate(result.question),
            "PASS" if result.naive_pass else "FAIL",
            f"{result.naive_time:.2f}",
            "PASS" if result.novel_pass else "FAIL",
            f"{result.novel_time:.2f}",
            result.filter_used,
        ])

    naive_pass_rate = (sum(1 for r in results if r.naive_pass) / len(results)) * 100
    novel_pass_rate = (sum(1 for r in results if r.novel_pass) / len(results)) * 100
    avg_naive_time = sum(r.naive_time for r in results) / len(results)
    avg_novel_time = sum(r.novel_time for r in results) / len(results)

    avg_row = [
        "PASS RATE",
        f"{naive_pass_rate:.0f}%",
        f"{avg_naive_time:.2f}",
        f"{novel_pass_rate:.0f}%",
        f"{avg_novel_time:.2f}",
        "-",
    ]

    widths = [max(len(headers[i]), max((len(row[i]) for row in rows), default=0), len(avg_row[i])) for i in range(len(headers))]

    def make_separator(fill: str = "-") -> str:
        return "+" + "+".join(fill * (w + 2) for w in widths) + "+"

    def make_row(values: list[str]) -> str:
        padded = [f" {values[i]:<{widths[i]}} " for i in range(len(values))]
        return "|" + "|".join(padded) + "|"

    print("\nFINAL EVALUATION SCORES (PASS/FAIL)")
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
        print(f"  Result: {'PASS' if ev_naive.success else 'FAIL'} | Time: {time_naive:.2f}s\n")
        
        # NOVEL
        print(f"Running Novel Hierarchical RAG...")
        res_novel, time_novel, filters = run_novel_rag(q)
        ans_nov = res_novel["answer"]
        ev_novel = cast(EvaluationScore, eval_chain.invoke({"question": q, "ground_truth": t, "answer": ans_nov}))
        _print_answer_block("Novel Answer", ans_nov)
        print(f"  Result: {'GOOD' if ev_novel.success else 'POOR'} | Time: {time_novel:.2f}s")
        print(f"  Filter Extracted: {filters}\n")

        results.append(
            EvalResult(
                question=q, naive_answer=ans_n, novel_answer=ans_nov,
                naive_pass=ev_naive.success, naive_time=time_naive,
                novel_pass=ev_novel.success, novel_time=time_novel,
                filter_used=str(filters.get("source", "None")),
            )
        )
    _print_score_table(results)

if __name__ == "__main__":
    evaluate()