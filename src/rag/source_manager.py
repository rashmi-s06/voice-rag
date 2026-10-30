import uuid
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any
from src.rag.msmarco_loader import MSMarcoPassage
from src.rag.file_parser import parse_uploaded_file, chunk_text

logger = logging.getLogger(__name__)

@dataclass
class DocumentSource:
    source_id: str
    name: str
    file_type: str
    num_chunks: int
    created_at: float
    passages: List[MSMarcoPassage] = field(default_factory=list)
    active: bool = True

class SourceManager:
    """NotebookLM-style source manager: allows users to curate their own knowledge base."""

    def __init__(self):
        self.sources: Dict[str, DocumentSource] = {}

    def add_source_file(self, file_path: str, display_name: Optional[str] = None) -> DocumentSource:
        """Parses an uploaded file (.pdf, .tsv, .csv, .txt, .md), extracts passages, and indexes into the notebook."""
        path = Path(file_path)
        name = display_name or path.name
        source_id = f"src_{uuid.uuid4().hex[:8]}"

        raw_passages = parse_uploaded_file(str(path))

        # Format passage IDs with explicit NotebookLM-style source provenance
        norm_passages: List[MSMarcoPassage] = []
        for idx, p in enumerate(raw_passages, 1):
            pid = f"{name}#Chunk{idx}"
            meta = dict(p.metadata) if p.metadata else {}
            meta.update({
                "source_id": source_id,
                "source_name": name,
                "chunk_index": idx
            })
            norm_passages.append(MSMarcoPassage(passage_id=pid, text=p.text, metadata=meta))

        source = DocumentSource(
            source_id=source_id,
            name=name,
            file_type=path.suffix.lower() or "file",
            num_chunks=len(norm_passages),
            created_at=time.time(),
            passages=norm_passages,
            active=True
        )
        self.sources[source_id] = source
        logger.info(f"Added source '{name}' ({len(norm_passages)} chunks) to notebook.")
        return source

    def add_source_text(self, text: str, title: str = "Pasted Note") -> DocumentSource:
        """Adds a direct user text note into the notebook knowledge base."""
        source_id = f"src_{uuid.uuid4().hex[:8]}"
        chunks = chunk_text(text)
        if not chunks:
            chunks = [text.strip()] if text.strip() else []

        passages: List[MSMarcoPassage] = []
        for idx, chunk in enumerate(chunks, 1):
            pid = f"{title}#Chunk{idx}"
            passages.append(MSMarcoPassage(
                passage_id=pid,
                text=chunk,
                metadata={"source_id": source_id, "source_name": title, "chunk_index": idx}
            ))

        source = DocumentSource(
            source_id=source_id,
            name=title,
            file_type="note",
            num_chunks=len(passages),
            created_at=time.time(),
            passages=passages,
            active=True
        )
        self.sources[source_id] = source
        logger.info(f"Added note '{title}' ({len(passages)} chunks) to notebook.")
        return source

    def remove_source(self, source_id: str) -> bool:
        """Removes a source document and its passages from the notebook."""
        if source_id in self.sources:
            removed = self.sources.pop(source_id)
            logger.info(f"Removed source '{removed.name}' from notebook.")
            return True
        return False

    def clear_all_sources(self):
        """Clears all sources from the notebook."""
        self.sources.clear()
        logger.info("Cleared all sources from notebook.")

    def get_all_active_passages(self) -> List[MSMarcoPassage]:
        """Returns all passages from currently active sources."""
        all_passages: List[MSMarcoPassage] = []
        for src in self.sources.values():
            if src.active:
                all_passages.extend(src.passages)
        return all_passages

    def get_sources_summary(self) -> List[Dict[str, Any]]:
        """Returns metadata list of all sources for UI display."""
        return [
            {
                "source_id": s.source_id,
                "name": s.name,
                "file_type": s.file_type,
                "num_chunks": s.num_chunks,
                "active": s.active,
                "created_at": s.created_at
            }
            for s in self.sources.values()
        ]

    def load_starter_sample(self) -> DocumentSource:
        """Loads a starter knowledge base (Tech & Science overview) for rapid 1-click testing."""
        starter_text = (
            "Artificial intelligence (AI) is transforming science, software engineering, and industry. "
            "Large language models utilize transformer architectures with self-attention mechanisms to process text tokens.\n\n"
            "Retrieval-Augmented Generation (RAG) combines dense vector search with generative models. "
            "By querying an external database or personal vector store, RAG grounds answers in verified facts, preventing hallucination.\n\n"
            "Mitochondria are membrane-bound cell organelles that generate chemical energy in the form of ATP. "
            "They are frequently referred to as the powerhouse of the cell in biological sciences.\n\n"
            "The speed of light in vacuum is defined as 299,792,458 metres per second (approx 300,000 km/s). "
            "It is denoted as c and represents the universal speed limit in modern physics.\n\n"
            "The Python programming language was created by Guido van Rossum and first released in 1991. "
            "It emphasizes code readability and programmer productivity with clean syntax."
        )
        return self.add_source_text(starter_text, title="Starter_Knowledge_Base.txt")
