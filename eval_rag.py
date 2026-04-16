import os
import textwrap
import time
from dataclasses import dataclass
from typing import Any, cast
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from rag_schema import PaperKey, PAPER_ALIASES, ROUTER_SYSTEM_PROMPT, SearchFilters

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

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_SYSTEM_PROMPT),
        ("user", "{query}"),
    ]
)

def run_naive_rag(query: str):
    start = time.time()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    chain = create_retrieval_chain(retriever, document_chain)
    ans = chain.invoke({"input": query})
    return ans, time.time() - start

def run_novel_rag(query: str):
    start = time.time()
    structured_llm = router_prompt | router_llm.with_structured_output(SearchFilters)
    extracted_filters = cast(SearchFilters, structured_llm.invoke({"query": query}))
    
    chroma_filter = {}
    if extracted_filters.target_source != PaperKey.unknown:
        mapped_source = PAPER_ALIASES.get(extracted_filters.target_source)
        if mapped_source:
            chroma_filter["source"] = mapped_source 
            
    search_kwargs: dict[str, Any] = {"k": 10}
    if chroma_filter:
        search_kwargs["filter"] = chroma_filter

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

    if chroma_filter:
        test_docs = retriever.invoke(query)
        if len(test_docs) == 0:
            chroma_filter = {}
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
        
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
