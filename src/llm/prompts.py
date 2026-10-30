from typing import List, Dict
from src.config import Config

STRICT_FACTUALITY_SYSTEM_PROMPT = f"""You are a strictly grounded, factual AI research assistant operating as a personal knowledge notebook (NotebookLM style).
You MUST follow these rules with zero exceptions:
1. ONLY use information explicitly stated in the provided User Notebook Sources below.
2. NEVER speculate, extrapolate, assume, or bring in external knowledge.
3. If the provided sources do NOT explicitly contain the answer, you MUST respond EXACTLY with:
   "{Config.STRICT_REFUSAL_MESSAGE}"
4. For EVERY fact or claim you state, you MUST cite the source in brackets using its exact source identifier, like: [Source: <filename>, Chunk #<id>].
5. Multilingual rule: If the user asks a question in Hindi (or another language), answer in that language, but strictly ground your answer only in the facts from the provided sources.
6. Keep answers concise, factual, and direct."""

def build_rag_prompt(query: str, retrieved_passages: List[Dict], language: str = "en") -> str:
    """Formats retrieved notebook passages and user query into a strictly grounded prompt."""
    if not retrieved_passages:
        return f"""User Query: {query}

Context:
No relevant sources or passages were found in the notebook.

Instructions: If no sources are present or relevant, output the exact strict refusal message:
"{Config.STRICT_REFUSAL_MESSAGE}"
"""

    context_blocks = []
    for p in retrieved_passages:
        pid = p.get("passage_id", "Unknown")
        meta = p.get("metadata", {})
        source_name = meta.get("source_name") or meta.get("source") or pid
        chunk_idx = meta.get("chunk_index", "")
        chunk_label = f", Chunk #{chunk_idx}" if chunk_idx else ""
        text = p.get("text", "").strip()
        context_blocks.append(f"--- [Source: {source_name}{chunk_label}] ---\n{text}")

    context_str = "\n\n".join(context_blocks)

    return f"""User's Notebook Reference Sources:
{context_str}

User Query: {query}
Detected Query Language: {language}

Answer the query strictly based on the notebook sources above. Always cite the exact source [Source: <name>, Chunk #<id>]. If the information is not present, respond with the strict refusal message."""
