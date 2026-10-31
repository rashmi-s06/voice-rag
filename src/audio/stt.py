import logging
import time
from pathlib import Path
from typing import Tuple, Optional
from src.config import Config

logger = logging.getLogger(__name__)

class SpeechToText:
    """High-speed automatic speech recognition with automatic language detection."""

    def __init__(self, model_size: str = Config.WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            logger.info(f"Initializing Faster-Whisper model ({self.model_size})...")
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=Config.WHISPER_DEVICE,
                compute_type=Config.WHISPER_COMPUTE_TYPE,
            )
            logger.info("Faster-Whisper model loaded.")
        return self._model

    def transcribe(self, audio_path: str) -> Tuple[str, str, float]:
        """Transcribes audio file to text.
        
        Returns:
            transcribed_text: Extracted text query.
            detected_language: ISO language code (e.g. 'en', 'hi').
            stt_latency_ms: Milliseconds taken for transcription.
        """
        start_time = time.perf_counter()
        model = self._get_model()

        segments, info = model.transcribe(
            audio_path,
            beam_size=1,  # Greedy search for minimum latency
            temperature=0.0,
            vad_filter=True,  # Voice activity detector cuts silence
        )

        text_pieces = [segment.text.strip() for segment in segments]
        transcribed_text = " ".join(text_pieces).strip()
        detected_language = info.language or "en"

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"Transcribed audio in {latency_ms:.1f}ms (Lang: {detected_language}): '{transcribed_text}'"
        )
        return transcribed_text, detected_language, latency_ms
