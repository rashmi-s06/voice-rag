# 🎙️ Voice-Enabled RAG System (MS MARCO Grounded)

An end-to-end, ultra-low latency, voice-enabled Retrieval-Augmented Generation (RAG) system built to the specifications of the Open Trials technical brief.

## 🌟 Key Architecture & Capabilities

1. **Voice-to-Voice Pipeline**:
   - **Speech-to-Text (STT)**: High-speed transcription using `faster-whisper` with automatic language detection.
   - **Text-to-Speech (TTS)**: Natural neural speech synthesis using Microsoft `edge-tts` with language-adaptive voices (`en-US-JennyNeural` for English, `hi-IN-SwaraNeural` for Hindi).
2. **Strict MS MARCO Grounding**:
   - Queries and retrieves from the MS MARCO passage dataset with verifiable `[Passage ID: <id>]` citations for every single factual claim.
3. **Zero Hallucination Guardrails**:
   - Built-in relevance confidence gate (`SIMILARITY_THRESHOLD = 0.28`). Out-of-domain queries trigger an immediate refusal without speculative fabrication:
     > *"The provided MS MARCO dataset does not contain sufficient information to answer this question."*
4. **Sub-200ms Latency Target**:
   - In-memory **FAISS** vector search (<5ms) fused with **BM25Okapi** keyword search via Reciprocal Rank Fusion (RRF).
   - Real-time latency telemetry measuring STT, retrieval, LLM Time-to-First-Token (TTFT), and total end-to-end execution.
5. **Multilingual Support (English & Hindi)**:
   - Uses `paraphrase-multilingual-MiniLM-L12-v2` dense embeddings, allowing cross-lingual queries (e.g. asking in Hindi and retrieving relevant English MS MARCO passages with Hindi grounded answers).

---

## 🚀 Quickstart

### 1. Prerequisites & Environment
The project is managed with `uv` and Python 3.12:

```bash
# Activate virtual environment (Windows PowerShell)
.venv\Scripts\activate
```

### 2. Configure API Key
Create a `.env` file or export your Gemini API key:
```bash
GEMINI_API_KEY="your-gemini-pro-api-key"
```
*(You can also paste the API key directly into the Web UI sidebar at runtime!)*

### 3. Launch Interactive Web App
```bash
python app.py
```
Open your browser to `http://127.0.0.1:7860`. You can:
- Speak directly into your microphone or upload audio files.
- Inspect the real-time latency telemetry breakdown.
- View the retrieved MS MARCO passages and similarity scores in the Evidence Inspector.

### 4. Run Automated Tests
```bash
python -m unittest test_pipeline.py
```

### 5. Run the Benchmark Suite
```bash
python benchmark.py
```
