import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from src.config import Config
from src.telemetry import LatencyProfile
from src.rag.source_manager import SourceManager, DocumentSource
from src.rag.msmarco_loader import MSMarcoPassage
from src.rag.vector_index import FaissVectorIndex
from src.rag.bm25_index import BM25Index
from src.rag.hybrid_retriever import HybridRetriever
from src.audio.stt import SpeechToText
from src.audio.tts import TextToSpeech
from src.llm.generator import GeminiGenerator

logger = logging.getLogger(__name__)

class VoiceRAGPipeline:
    """NotebookLM-style Voice & Text RAG Pipeline grounded in user-curated sources."""

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.vector_index = FaissVectorIndex()
        self.bm25_index = BM25Index()
        self.source_manager = SourceManager()
        self.retriever = HybridRetriever(self.vector_index, self.bm25_index)
        self.llm = GeminiGenerator(api_key=gemini_api_key)
        self.is_initialized = False

    def initialize(self, load_starter_demo: bool = False) -> None:
        """Initializes the pipeline. If load_starter_demo is True, loads a starter sample knowledge base."""
        logger.info("Initializing NotebookLM Voice RAG Pipeline...")
        if load_starter_demo:
            self.load_starter_pack()
        else:
            self._reindex()
        self.is_initialized = True
        logger.info("NotebookLM Voice RAG Pipeline initialization complete.")

    def _reindex(self) -> int:
        """Internal helper to rebuild FAISS and BM25 indices from current active sources."""
        passages = self.source_manager.get_all_active_passages()
        self.vector_index.build_index(passages)
        self.bm25_index.build_index(passages)
        self.retriever = HybridRetriever(self.vector_index, self.bm25_index)
        return len(passages)

    def add_source_file(self, file_path: str, display_name: Optional[str] = None) -> Tuple[DocumentSource, int]:
        """Adds a user-uploaded file (.pdf, .tsv, .csv, .txt, .md) to the notebook and re-indexes."""
        source = self.source_manager.add_source_file(file_path, display_name)
        total_chunks = self._reindex()
        return source, total_chunks

    def add_source_text(self, text: str, title: str = "Pasted Note") -> Tuple[DocumentSource, int]:
        """Adds a direct text note to the notebook and re-indexes."""
        source = self.source_manager.add_source_text(text, title)
        total_chunks = self._reindex()
        return source, total_chunks

    def remove_source(self, source_id: str) -> int:
        """Removes a source document from the notebook and updates index."""
        self.source_manager.remove_source(source_id)
        return self._reindex()

    def clear_all_sources(self) -> int:
        """Clears all sources from the notebook."""
        self.source_manager.clear_all_sources()
        return self._reindex()

    def load_starter_pack(self) -> Tuple[DocumentSource, int]:
        """Loads starter sample knowledge base for 1-click test exploration."""
        source = self.source_manager.load_starter_sample()
        total_chunks = self._reindex()
        return source, total_chunks

    def get_sources_summary(self) -> List[Dict[str, Any]]:
        """Returns metadata list of all sources in the notebook."""
        return self.source_manager.get_sources_summary()

    def get_total_passages_count(self) -> int:
        """Returns total active chunks/passages in the notebook."""
        return len(self.source_manager.get_all_active_passages())

    def run(
        self,
        audio_path: Optional[str] = None,
        text_query: Optional[str] = None,
        synthesize_voice: bool = True
    ) -> Tuple[str, Optional[str], LatencyProfile, List[Dict]]:
        """Processes voice or text query end-to-end with full latency telemetry.
        
        Returns:
            answer_text: Grounded response with citations.
            audio_out_path: Path to generated spoken audio file (or None).
            profile: LatencyProfile containing millisecond breakdown.
            retrieved_passages: Top notebook passages retrieved.
        """
        if not self.is_initialized:
            self.initialize()

        pipeline_start = time.perf_counter()
        profile = LatencyProfile()

        query = ""
        lang = "en"

        # --- Stage 1: Speech-to-Text ---
        if audio_path:
            query, lang, stt_ms = self.stt.transcribe(audio_path)
            profile.stt_ms = stt_ms
            profile.language = lang
        elif text_query:
            query = text_query.strip()
            # Heuristic language check for Devanagari script (Hindi)
            if any("\u0900" <= ch <= "\u097f" for ch in query):
                lang = "hi"
            profile.language = lang
        else:
            return "No voice or text input received.", None, profile, []

        if not query:
            return "Could not transcribe any speech from the audio.", None, profile, []

        # --- Check for Empty Notebook ---
        total_passages = self.get_total_passages_count()
        if total_passages == 0:
            profile.refusal_triggered = True
            empty_msg = (
                "Your notebook is empty. Please upload source documents (PDF, CSV, TSV, TXT, MD) "
                "or paste notes in the Source Drawer to begin asking questions."
            )
            audio_out = None
            if synthesize_voice:
                audio_out, tts_ms = self.tts.synthesize(empty_msg, language=lang)
                profile.tts_ms = tts_ms
            profile.total_ms = (time.perf_counter() - pipeline_start) * 1000.0
            return empty_msg, audio_out, profile, []

        # --- Stage 2: Hybrid Retrieval & Anti-Hallucination Gate ---
        retrieval_start = time.perf_counter()
        retrieved_passages, max_score, passed_guardrail = self.retriever.retrieve(query)
        profile.retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0
        profile.retrieved_count = len(retrieved_passages)
        profile.passed_guardrail = passed_guardrail

        # --- Stage 3: Anti-Hallucination Guardrail Check ---
        if not passed_guardrail:
            profile.refusal_triggered = True
            answer_text = Config.STRICT_REFUSAL_MESSAGE
            profile.llm_ttft_ms = 0.0
            profile.llm_total_ms = 0.0
        else:
            # --- Stage 4: LLM Generation (Grounded) ---
            answer_text, ttft_ms, total_gen_ms = self.llm.generate_grounded_answer(
                query=query,
                retrieved_passages=retrieved_passages,
                language=lang
            )
            profile.llm_ttft_ms = ttft_ms
            profile.llm_total_ms = total_gen_ms

        # --- Stage 5: Text-to-Speech Synthesis ---
        audio_out = None
        if synthesize_voice and answer_text:
            audio_out, tts_ms = self.tts.synthesize(answer_text, language=lang)
            profile.tts_ms = tts_ms

        profile.total_ms = (time.perf_counter() - pipeline_start) * 1000.0

        return answer_text, audio_out, profile, retrieved_passages
