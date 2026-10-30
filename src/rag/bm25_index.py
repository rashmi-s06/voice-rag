import re
import logging
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from src.rag.msmarco_loader import MSMarcoPassage
from src.config import Config

logger = logging.getLogger(__name__)

# Common stop words in English to prevent trivial matches on function words
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", 
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", 
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", 
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", 
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him", 
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its", 
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", 
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", 
    "ourselves", "out", "over", "own", "same", "shan't", "she", "should", "shouldn't", 
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them", 
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", 
    "too", "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't", 
    "what", "when", "where", "which", "while", "who", "whom", "why", "with", "won't", 
    "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves"
}

def tokenize(text: str) -> List[str]:
    """Tokenizes text into content terms, removing standard stop words."""
    words = re.findall(r"\w+", text.lower())
    filtered = [w for w in words if w not in STOP_WORDS and len(w) > 1]
    return filtered if filtered else words  # Fallback to raw words if all were filtered

class BM25Index:
    """Sparse keyword index using BM25Okapi for exact content term retrieval."""

    def __init__(self):
        self.bm25: BM25Okapi = None
        self.passages: List[MSMarcoPassage] = []

    def build_index(self, passages: List[MSMarcoPassage]) -> None:
        """Tokenizes passage texts and initializes BM25Okapi index."""
        self.passages = passages
        if not passages:
            self.bm25 = None
            logger.info("Cleared BM25 index (0 passages).")
            return
        corpus = [tokenize(p.text) for p in passages]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"Built stop-word-filtered BM25 index over {len(passages)} passages.")

    def search(self, query: str, top_k: int = Config.TOP_K_RETRIEVAL) -> List[Tuple[MSMarcoPassage, float]]:
        """Searches BM25 index and returns list of (passage, bm25_score)."""
        if not self.bm25 or not self.passages:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        raw_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = float(raw_scores[idx])
            if score > 0.5:  # Non-trivial keyword match threshold
                results.append((self.passages[idx], score))
        return results
