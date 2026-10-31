from typing import List, Dict
from src.config import Config

STRICT_FACTUALITY_SYSTEM_PROMPT = f"""You are a strictly grounded, factual question-answering assistant for an evaluation benchmark.
You MUST follow these rules with zero exceptions:
1. ONLY use information explicitly stated in the provided MS MARCO context passages below.
2. NEVER speculate, extrapolate, assume, or bring in external knowledge.
3. If the provided passages do NOT explicitly contain the answer, you MUST respond EXACTLY with:
   "{Config.STRICT_REFUSAL_MESSAGE}"
4. For EVERY fact or claim you state, you MUST cite the source passage ID in brackets, like: [Passage ID: 100001].
5. Multilingual rule: If the user asks a question in Hindi (or another language), answer in that language, but strictly ground your answer only in the facts from the provided English passages.
6. Keep answers concise, factual, and direct."""

def build_rag_prompt(query: str, retrieved_passages: List[Dict], language: str = "en") -> str:
    """Formats retrieved MS MARCO passages and user query into a strictly grounded prompt."""
    if not retrieved_passages:
        return f"""User Query: {query}

Context:
No relevant passages were found in the MS MARCO dataset.

Instructions: Answer the query using ONLY the context above. If not found, output the strict refusal message."""

    context_blocks = []
    for p in retrieved_passages:
        pid = p.get("passage_id", "Unknown")
        text = p.get("text", "").strip()
        context_blocks.append(f"--- [Passage ID: {pid}] ---\n{text}")

    context_str = "\n\n".join(context_blocks)

    return f"""MS MARCO Reference Passages:
{context_str}

User Query: {query}
Detected Query Language: {language}

Answer the query strictly based on the passages above. Always cite [Passage ID: <id>]. If the information is not present in the passages, refuse as instructed."""
