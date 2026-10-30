import asyncio
import logging
import time
import tempfile
from pathlib import Path
from typing import Tuple
import edge_tts
from src.config import Config

logger = logging.getLogger(__name__)

class TextToSpeech:
    """Fast neural Text-to-Speech using edge-tts with language-adaptive voices."""

    def __init__(self):
        self.voice_en = Config.DEFAULT_VOICE_EN
        self.voice_hi = Config.DEFAULT_VOICE_HI

    def _select_voice(self, language: str) -> str:
        lang = (language or "en").lower()
        if lang.startswith("hi"):
            return self.voice_hi
        return self.voice_en

    async def _synthesize_async(self, text: str, voice: str, output_path: str) -> None:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)

    def synthesize(self, text: str, language: str = "en") -> Tuple[str, float]:
        """Synthesizes text into audio file (.mp3).
        
        Returns:
            output_audio_path: Path to generated audio file.
            tts_latency_ms: Milliseconds taken for synthesis.
        """
        start_time = time.perf_counter()
        voice = self._select_voice(language)

        # Truncate or clean citations from speech for natural audio flow
        speech_text = text.replace("[Passage ID:", "Reference passage")
        if not speech_text.strip():
            speech_text = "No response available."

        temp_dir = Path(tempfile.gettempdir()) / "voice_rag_audio"
        temp_dir.mkdir(parents=True, exist_ok=True)
        out_file = temp_dir / f"response_{int(time.time() * 1000)}.mp3"

        try:
            asyncio.run(self._synthesize_async(speech_text, voice, str(out_file)))
        except RuntimeError:
            # Fallback if an event loop is already active in current thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._synthesize_async(speech_text, voice, str(out_file)))
            loop.close()

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"Synthesized audio in {latency_ms:.1f}ms with voice {voice}")
        return str(out_file), latency_ms
