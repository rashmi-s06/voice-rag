import logging
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from src.rag.msmarco_loader import MSMarcoPassage
from src.rag.vector_index import FaissVectorIndex
from src.rag.bm25_index import BM25Index
from src.config import Config

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Combines Dense Vector Search (FAISS) and Sparse Keyword Search (BM25)
    using Reciprocal Rank Fusion (RRF), equipped with a strict anti-hallucination
    confidence gate with cross-lingual calibration.
    """

    def __init__(self, vector_index: FaissVectorIndex, bm25_index: BM25Index):
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.rrf_k = Config.RRF_K
        self.bm25_keyword_threshold = 2.0

    def retrieve(
        self, 
        query: str, 
        top_k: int = Config.TOP_K_RETRIEVAL
    ) -> Tuple[List[Dict], float, bool]:
        """Executes hybrid retrieval with anti-hallucination filtering.
        
        Returns:
            ranked_results: List of dicts containing passage info and fused scores.
            max_dense_score: Highest dense cosine similarity observed.
            passed_guardrail: Whether the query exceeded the relevance threshold.
        """
        # 1. Dense retrieval
        dense_hits = self.vector_index.search(query, top_k=top_k * 2)
        # 2. Sparse retrieval
        bm25_hits = self.bm25_index.search(query, top_k=top_k * 2)

        max_dense_score = dense_hits[0][1] if dense_hits else 0.0
        max_bm25_score = bm25_hits[0][1] if bm25_hits else 0.0

        # Calibrated threshold:
        # Cross-lingual queries (e.g. Hindi -> English passages) naturally align around 0.33-0.55
        # Monolingual English queries align >0.55 (spurious trick queries are <0.32)
        is_cross_lingual = any(ord(c) > 127 for c in query)
        effective_threshold = 0.32 if is_cross_lingual else 0.36

        # Anti-Hallucination Guardrail Check:
        # Requires either solid semantic relevance or valid keyword match in MS MARCO corpus
        passed_guardrail = (max_dense_score >= effective_threshold) or (max_bm25_score >= self.bm25_keyword_threshold)

        if not passed_guardrail:
            logger.warning(
                f"Query '{query}' failed relevance threshold (Dense: {max_dense_score:.3f} < {effective_threshold}, "
                f"BM25: {max_bm25_score:.2f} < {self.bm25_keyword_threshold}). Triggering strict refusal guardrail."
            )
            return [], max_dense_score, False

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = defaultdict(float)
        passage_map: Dict[str, MSMarcoPassage] = {}
        dense_score_map: Dict[str, float] = {}
        bm25_score_map: Dict[str, float] = {}

        for rank, (passage, score) in enumerate(dense_hits):
            pid = passage.passage_id
            passage_map[pid] = passage
            dense_score_map[pid] = score
            rrf_scores[pid] += 1.0 / (self.rrf_k + rank + 1)

        for rank, (passage, score) in enumerate(bm25_hits):
            pid = passage.passage_id
            passage_map[pid] = passage
            bm25_score_map[pid] = score
            rrf_scores[pid] += 1.0 / (self.rrf_k + rank + 1)

        # Sort candidate passages by combined RRF score
        sorted_pids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)[:top_k]

        results = []
        for pid in sorted_pids:
            p = passage_map[pid]
            results.append({
                "passage_id": p.passage_id,
                "text": p.text,
                "metadata": p.metadata,
                "rrf_score": rrf_scores[pid],
                "dense_score": dense_score_map.get(pid, 0.0),
                "bm25_score": bm25_score_map.get(pid, 0.0),
            })

        return results, max_dense_score, True
