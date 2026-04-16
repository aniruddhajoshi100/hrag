from enum import Enum

from pydantic import BaseModel, Field


class PaperKey(str, Enum):
    qlora = "qlora"
    attention = "attention"
    chain_of_thought = "chain_of_thought"
    unknown = "unknown"


PAPER_ALIASES = {
    PaperKey.attention: "p1.pdf",
    PaperKey.qlora: "p2.pdf",
    PaperKey.chain_of_thought: "p3.pdf",
}


ROUTER_SYSTEM_PROMPT = (
    "You are a routing assistant. Map the query to exactly one paper key.\n"
    "Use qlora for the QLoRA paper (keywords: qlora, guanaco, 4-bit, nf4, lora).\n"
    "Use attention for the Attention Is All You Need / Transformer paper "
    "(keywords: attention is all you need, transformer, self-attention).\n"
    "Use chain_of_thought for the Chain-of-Thought prompting paper "
    "(keywords: chain of thought, chain-of-thought, cot).\n"
    "If the query does not target a specific paper, use unknown."
)


class SearchFilters(BaseModel):
    """Schema for extracting metadata filters from a user's natural language query."""

    target_source: PaperKey = Field(
        default=PaperKey.unknown,
        description=(
            "Identify which specific paper is being asked about. "
            "Select from the provided enum list. If no specific paper is mentioned, "
            "select 'unknown'."
        ),
    )
    target_section: str | None = Field(
        default=None,
        description="The specific section of the paper mentioned. If no section is mentioned, return None.",
    )
