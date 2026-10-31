import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from src.config import Config
from src.telemetry import LatencyProfile
from src.rag.msmarco_loader import MSMarcoLoader, MSMarcoPassage
from src.rag.vector_index import FaissVectorIndex
from src.rag.bm25_index import BM25Index
from src.rag.hybrid_retriever import HybridRetriever
from src.audio.stt import SpeechToText
from src.audio.tts import TextToSpeech
from src.llm.generator import GeminiGenerator

logger = logging.getLogger(__name__)

class VoiceRAGPipeline:
    """End-to-End Voice-Enabled RAG Pipeline strictly grounded in MS MARCO."""

    def __init__(self, data_path: Optional[Path] = None, gemini_api_key: Optional[str] = None):
        self.data_path = data_path or Config.SAMPLE_MSMARCO_PATH
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.vector_index = FaissVectorIndex()
        self.bm25_index = BM25Index()
        self.retriever: Optional[HybridRetriever] = None
        self.llm = GeminiGenerator(api_key=gemini_api_key)
        self.is_initialized = False

    def initialize(self, force_reindex: bool = False) -> None:
        """Initializes knowledge base from MS MARCO dataset and prepares indices."""
        logger.info("Initializing Voice RAG Pipeline...")

        # 1. Ensure sample dataset exists if using default path
        if self.data_path == Config.SAMPLE_MSMARCO_PATH:
            MSMarcoLoader.ensure_sample_tsv(self.data_path)

        # 2. Load passages
        passages = MSMarcoLoader.load_tsv(self.data_path)

        # 3. Check cached FAISS index
        cached = False
        if not force_reindex and Config.FAISS_INDEX_PATH.exists() and Config.PASSAGES_CACHE_PATH.exists():
            cached = self.vector_index.load(Config.FAISS_INDEX_PATH, Config.PASSAGES_CACHE_PATH)

        if not cached:
            self.vector_index.build_index(passages)
            self.vector_index.save(Config.FAISS_INDEX_PATH, Config.PASSAGES_CACHE_PATH)

        # 4. Build BM25 index
        self.bm25_index.build_index(passages)

        # 5. Connect Hybrid Retriever
        self.retriever = HybridRetriever(self.vector_index, self.bm25_index)
        self.is_initialized = True
        logger.info("Voice RAG Pipeline initialization complete.")

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
            retrieved_passages: Top MS MARCO passages retrieved.
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
            # Fast heuristic language check for Devanagari script (Hindi)
            if any("\u0900" <= ch <= "\u097f" for ch in query):
                lang = "hi"
            profile.language = lang
        else:
            return "No voice or text input received.", None, profile, []

        if not query:
            return "Could not transcribe any speech from the audio.", None, profile, []

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
