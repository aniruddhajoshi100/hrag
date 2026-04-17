import os
import textwrap
import time
from typing import Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import EnsembleRetriever
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
    data = vectorstore.get(include=["metadatas", "documents"])
    unique_titles = list(set(meta.get("title") for meta in data["metadatas"] if meta and "title" in meta))
    corpus_docs: list[Document] = []
    for raw_content, raw_meta in zip(data.get("documents", []), data.get("metadatas", [])):
        if not raw_content or not str(raw_content).strip():
            continue
        corpus_docs.append(Document(page_content=str(raw_content), metadata=raw_meta or {}))
except Exception:
    unique_titles = []
    corpus_docs = []

if not unique_titles:
    print("⚠️ WARNING: No titles found in the database. Did you re-ingest your data with the new chunks.json?")

llm = ChatGroq(temperature=0.0, api_key=SecretStr(groq_api_key), model="llama-3.3-70b-versatile")


def _infer_subsection_from_section(section_value: str) -> tuple[str, str | None]:
    text = str(section_value or "").strip()
    if ":" not in text:
        return text, None

    prefix, suffix = text.split(":", 1)
    prefix = prefix.strip()
    suffix = suffix.strip()
    if not prefix or not suffix:
        return text, None

    if len(prefix.split()) <= 8:
        return prefix, suffix

    return text, None


def ensure_hierarchy_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    level_1 = normalized.get("path_level_1") or normalized.get("title") or normalized.get("source") or "Unknown Document"
    level_2 = normalized.get("path_level_2") or normalized.get("section") or "Main Text"
    level_3 = normalized.get("path_level_3") or normalized.get("subsection")

    inferred_section, inferred_subsection = _infer_subsection_from_section(level_2)
    level_2 = inferred_section or "Main Text"
    if not level_3:
        level_3 = inferred_subsection

    normalized["path_level_1"] = str(level_1)
    normalized["path_level_2"] = str(level_2)
    normalized["section"] = normalized["path_level_2"]

    hierarchy_parts = [normalized["path_level_1"], normalized["path_level_2"]]
    if level_3 and str(level_3).strip():
        normalized["path_level_3"] = str(level_3)
        normalized["subsection"] = str(level_3)
        hierarchy_parts.append(normalized["path_level_3"])
    else:
        normalized.pop("path_level_3", None)
        if "subsection" in normalized and not str(normalized.get("subsection", "")).strip():
            normalized.pop("subsection", None)

    normalized["hierarchy_path"] = " -> ".join(hierarchy_parts)
    normalized["path_depth"] = len(hierarchy_parts)
    return normalized


for doc in corpus_docs:
    doc.metadata = ensure_hierarchy_metadata(doc.metadata)


def _matches_filter(metadata: dict[str, Any], chroma_filter: dict[str, Any]) -> bool:
    if not chroma_filter:
        return True
    for key, expected in chroma_filter.items():
        if isinstance(expected, dict) and "$eq" in expected:
            expected = expected["$eq"]
        if metadata.get(key) != expected:
            return False
    return True


def build_hybrid_retriever(vectorstore, corpus_docs: list[Document], search_kwargs: dict[str, Any]):
    vector_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    if not corpus_docs:
        return vector_retriever

    bm25_candidates = [doc for doc in corpus_docs if _matches_filter(doc.metadata, search_kwargs.get("filter", {}))]
    if not bm25_candidates:
        bm25_candidates = corpus_docs

    bm25_retriever = BM25Retriever.from_documents(bm25_candidates)
    bm25_retriever.k = int(search_kwargs.get("k", 10))

    return EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.7, 0.3])

# --- 3. NOVEL RAG CHAIN BUILDER ---
def build_chain(llm, vectorstore, corpus_docs, search_kwargs):
    prompt = ChatPromptTemplate.from_template(
        """You are a highly precise academic research assistant. Use ONLY the following context from academic papers to answer the query. 
        Each piece of context includes its hierarchical path (Title -> Section -> Subsection, when available). Pay close attention to this path to understand where the information comes from.
        
        If you cannot find the answer in the context, strictly output "Insufficient data to answer this query based on the retrieved context." Do not hallucinate.
        
        Context:
        {context}
        
        Question: {input}
        
        Answer:"""
    )
    
    # Updated to use {title} instead of {source}
    document_prompt = PromptTemplate(
        input_variables=["page_content", "hierarchy_path"],
        template="[PATH: {hierarchy_path}]\n{page_content}\n"
    )

    retriever = build_hybrid_retriever(vectorstore, corpus_docs, search_kwargs)
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
        
    chain = build_chain(llm, vectorstore, corpus_docs, search_kwargs)
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


def _print_retrieved_paths(title: str, response_payload: dict[str, Any]) -> None:
    print(f"{title}:")
    docs = response_payload.get("context", []) if isinstance(response_payload, dict) else []
    if not docs:
        print("  [no retrieved context]")
        return

    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        path_value = metadata.get("hierarchy_path")
        if not path_value:
            title_value = metadata.get("title") or metadata.get("path_level_1") or "Unknown Title"
            section_value = metadata.get("section") or metadata.get("path_level_2") or "Main Text"
            subsection_value = metadata.get("path_level_3") or metadata.get("subsection")
            path_parts = [str(title_value), str(section_value)]
            if subsection_value:
                path_parts.append(str(subsection_value))
            path_value = " -> ".join(path_parts)

        print(f"  {idx}. {path_value}")

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
        _print_retrieved_paths("Naive Retrieved Paths", res_naive)
        print(f"  Time: {time_naive:.2f}s\n")
        
        # NOVEL
        res_novel, time_novel, filters = run_novel_rag(q)
        ans_nov = res_novel["answer"]
        _print_answer_block("Novel Answer", ans_nov)
        _print_retrieved_paths("Novel Retrieved Paths", res_novel)
        print(f"  Filter Extracted: {filters}")
        print(f"  Time: {time_novel:.2f}s\n")
        print("=" * 80)

if __name__ == "__main__":
    evaluate()