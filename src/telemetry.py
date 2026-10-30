import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class LatencyProfile:
    stt_ms: float = 0.0
    retrieval_ms: float = 0.0
    llm_ttft_ms: float = 0.0
    llm_total_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0
    language: str = "en"
    retrieved_count: int = 0
    passed_guardrail: bool = True
    refusal_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "STT Latency (ms)": round(self.stt_ms, 2),
            "Retrieval & RRF (ms)": round(self.retrieval_ms, 2),
            "LLM Time to First Token (TTFT ms)": round(self.llm_ttft_ms, 2),
            "LLM Total Stream (ms)": round(self.llm_total_ms, 2),
            "TTS Synthesis (ms)": round(self.tts_ms, 2),
            "Total Pipeline E2E (ms)": round(self.total_ms, 2),
            "Detected Language": self.language,
            "Passages Retrieved": self.retrieved_count,
            "Guardrail Passed": self.passed_guardrail,
            "Refusal Triggered": self.refusal_triggered,
        }

    def summary_markdown(self) -> str:
        benchmark_status = "[Benchmark Target Met: Sub-200ms]" if self.total_ms <= 200 else "[Pipeline Latency Profile]"
        guardrail_label = "Passed (Grounded)" if self.passed_guardrail else "Refusal Triggered (Blocked)"
        return f"""#### Performance Metrics & Latency
| Metric | Value |
| :--- | :--- |
| **Speech-to-Text (STT)** | `{self.stt_ms:.1f} ms` |
| **Hybrid Retrieval & RRF** | `{self.retrieval_ms:.1f} ms` |
| **LLM Time-to-First-Token (TTFT)** | `{self.llm_ttft_ms:.1f} ms` |
| **LLM Generation Total** | `{self.llm_total_ms:.1f} ms` |
| **TTS Audio Synthesis** | `{self.tts_ms:.1f} ms` |
| **Total Pipeline Latency** | **`{self.total_ms:.1f} ms`** |
| **Detected Query Language** | `{self.language.upper()}` |
| **Guardrail Status** | `{guardrail_label}` |
| **Target Evaluation** | `{benchmark_status}` |
"""

class Timer:
    """Convenience context manager for high-precision latency profiling."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        self.duration_ms = (self.end - self.start) * 1000.0
