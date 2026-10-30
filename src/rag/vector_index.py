import json
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.rag.msmarco_loader import MSMarcoPassage
from src.config import Config

logger = logging.getLogger(__name__)

class FaissVectorIndex:
    """Ultra-fast, in-memory FAISS vector index with multilingual embeddings."""

    def __init__(self, model_name: str = Config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}...")
        self.encoder = SentenceTransformer(model_name)
        if hasattr(self.encoder, "get_embedding_dimension"):
            self.dim = self.encoder.get_embedding_dimension()
        else:
            self.dim = self.encoder.get_sentence_embedding_dimension()
        
        # Inner product index with normalized vectors is equivalent to Cosine Similarity
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dim)
        self.passages: List[MSMarcoPassage] = []

    def clear(self) -> None:
        """Clears all passages and resets FAISS index."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.passages = []
        logger.info("Cleared FAISS vector index.")

    def build_index(self, passages: List[MSMarcoPassage]) -> None:
        """Encodes passages and adds normalized embeddings to the FAISS index."""
        if not passages:
            self.clear()
            return

        logger.info(f"Encoding {len(passages)} passages into FAISS...")
        texts = [p.text for p in passages]
        
        start_time = time.perf_counter()
        embeddings = self.encoder.encode(
            texts, 
            batch_size=64, 
            show_progress_bar=False, 
            normalize_embeddings=True
        ).astype(np.float32)
        encode_time = (time.perf_counter() - start_time) * 1000.0

        # Reset and build FAISS index
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.passages = passages
        logger.info(f"Indexed {len(passages)} passages in {encode_time:.1f}ms. Total index size: {self.index.ntotal}")

    def search(self, query: str, top_k: int = Config.TOP_K_RETRIEVAL) -> List[Tuple[MSMarcoPassage, float]]:
        """Performs cosine similarity search against FAISS index.
        Returns list of (passage, cosine_similarity_score).
        """
        if self.index.ntotal == 0:
            return []

        query_vector = self.encoder.encode(
            [query], 
            normalize_embeddings=True
        ).astype(np.float32)

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, actual_k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx != -1 and idx < len(self.passages):
                results.append((self.passages[idx], float(score)))

        return results

    def save(self, index_file: Path, cache_file: Path) -> None:
        """Saves FAISS index and metadata cache to disk."""
        index_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_file))
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump([p.to_dict() for p in self.passages], f, ensure_ascii=False, indent=2)
        logger.info(f"Saved FAISS index to {index_file}")

    def load(self, index_file: Path, cache_file: Path) -> bool:
        """Loads FAISS index and metadata cache from disk if available."""
        if index_file.exists() and cache_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.passages = [MSMarcoPassage(**item) for item in data]
                logger.info(f"Loaded FAISS index with {self.index.ntotal} items from disk.")
                return True
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}")
        return False
