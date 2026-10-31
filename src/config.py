import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

class Config:
    # --- Paths ---
    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    SAMPLE_MSMARCO_PATH = DATA_DIR / "msmarco_sample.tsv"
    FAISS_INDEX_PATH = DATA_DIR / "faiss_index.bin"
    PASSAGES_CACHE_PATH = DATA_DIR / "passages_cache.json"

    # --- LLM Settings (Gemini) ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # Ultra-fast, highly grounded
    FALLBACK_MODEL = "gemini-1.5-flash"
    TEMPERATURE = 0.0  # Zero temperature for strictly factual, deterministic answers

    # --- Speech-to-Text (STT) Settings ---
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
    WHISPER_DEVICE = "cpu"
    WHISPER_COMPUTE_TYPE = "int8"

    # --- Text-to-Speech (TTS) Settings ---
    DEFAULT_VOICE_EN = "en-US-JennyNeural"
    DEFAULT_VOICE_HI = "hi-IN-SwaraNeural"

    # --- Embeddings & Retrieval Settings ---
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM = 384
    TOP_K_RETRIEVAL = 5
    RRF_K = 60  # Reciprocal Rank Fusion constant

    # --- Anti-Hallucination & Factuality Thresholds ---
    # Calibrated cosine similarity threshold for zero-hallucination guardrail
    SIMILARITY_THRESHOLD = 0.38
    STRICT_REFUSAL_MESSAGE = (
        "The provided MS MARCO dataset does not contain sufficient information to answer this question."
    )

    # --- Latency Benchmarking Target ---
    TARGET_TOTAL_LATENCY_MS = 200.0
