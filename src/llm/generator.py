import logging
import time
from typing import List, Dict, Tuple, Generator, Optional
from src.config import Config
from src.llm.prompts import STRICT_FACTUALITY_SYSTEM_PROMPT, build_rag_prompt

logger = logging.getLogger(__name__)

class GeminiGenerator:
    """Interfaces with Google Gemini for strictly grounded answers with TTFT tracking."""

    def __init__(self, api_key: Optional[str] = None, model: str = Config.GEMINI_MODEL):
        self.api_key = api_key or Config.GEMINI_API_KEY
        self.model = model
        self.client = None
        if self.api_key:
            self._init_client()

    def _init_client(self):
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Initialized Google GenAI client with model {self.model}.")
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI client: {e}")
            self.client = None

    def set_api_key(self, api_key: str):
        """Allows dynamically updating API key from UI or runtime."""
        self.api_key = api_key.strip()
        self._init_client()

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_passages: List[Dict],
        language: str = "en"
    ) -> Tuple[str, float, float]:
        """Generates grounded answer using streaming to capture Time-To-First-Token (TTFT).
        
        Returns:
            full_response_text: Completed answer.
            ttft_ms: Time in milliseconds to receive the first token.
            total_gen_ms: Total LLM response time in milliseconds.
        """
        prompt = build_rag_prompt(query, retrieved_passages, language=language)

        if not self.client:
            # If no API key configured, provide a clear direct-passage extraction fallback
            return self._fallback_grounded_response(query, retrieved_passages, language)

        start_time = time.perf_counter()
        first_token_time = None
        response_chunks = []

        try:
            from google.genai import types

            response_stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=STRICT_FACTUALITY_SYSTEM_PROMPT,
                    temperature=Config.TEMPERATURE,
                    max_output_tokens=350,
                )
            )

            for chunk in response_stream:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                if chunk.text:
                    response_chunks.append(chunk.text)

            end_time = time.perf_counter()
            ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else (end_time - start_time) * 1000.0
            total_gen_ms = (end_time - start_time) * 1000.0

            full_text = "".join(response_chunks).strip()
            return full_text, ttft_ms, total_gen_ms

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            # Try fallback model if 2.5-flash failed
            if self.model != Config.FALLBACK_MODEL:
                try:
                    logger.info(f"Retrying with fallback model {Config.FALLBACK_MODEL}...")
                    self.model = Config.FALLBACK_MODEL
                    return self.generate_grounded_answer(query, retrieved_passages, language)
                except Exception:
                    pass
            return self._fallback_grounded_response(query, retrieved_passages, language)

    def _fallback_grounded_response(
        self, 
        query: str, 
        retrieved_passages: List[Dict], 
        language: str
    ) -> Tuple[str, float, float]:
        """Failsafe response when API key is missing or offline, ensuring strict refusal if no passages."""
        start = time.perf_counter()
        if not retrieved_passages:
            text = Config.STRICT_REFUSAL_MESSAGE
        else:
            top_p = retrieved_passages[0]
            pid = top_p["passage_id"]
            p_text = top_p["text"]
            text = f"[Grounded Mode] Based on MS MARCO passage [Passage ID: {pid}]: {p_text}\n\n*(Note: Add your GEMINI_API_KEY to enable full live generative synthesis)*"
        duration = (time.perf_counter() - start) * 1000.0
        return text, duration, duration
