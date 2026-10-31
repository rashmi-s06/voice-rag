---
title: Voice-Enabled RAG System
colorFrom: pink
colorTo: green
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
---

# Voice-Enabled RAG System

- **Live GitHub Pages App**: [https://saniya-cps.github.io/voice-rag/](https://saniya-cps.github.io/voice-rag/)
- **Repository**: [https://github.com/saniya-cps/voice-rag](https://github.com/saniya-cps/voice-rag)

An end-to-end, ultra-low latency **Voice-Enabled Retrieval-Augmented Generation (RAG) System** in Python. Users curate their own personal source database (`.pdf`, `.tsv`, `.csv`, `.txt`, `.md`, or custom research notes) and interact via spoken voice or text in English and Hindi.

## System Architecture & Technical Pillars

1. **Voice-to-Voice Pipeline**:
   - **Speech-to-Text (STT)**: High-speed transcription using `faster-whisper` with automatic language detection.
   - **Text-to-Speech (TTS)**: Natural neural speech synthesis using Microsoft `edge-tts` with language-adaptive voices (`en-US-JennyNeural` for English, `hi-IN-SwaraNeural` for Hindi).
2. **Personal Source Knowledge Base**:
   - Upload multiple documents (`.pdf`, `.tsv`, `.csv`, `.txt`, `.md`) or paste custom notes.
   - Automatic paragraph and sentence chunking with verifiable `[Source: <filename>, Chunk #<id>]` citations for every factual claim.
3. **Zero-Hallucination Guardrails**:
   - Dual semantic and keyword relevance gate. Empty notebooks or out-of-domain queries trigger an immediate refusal without speculative fabrication:
     > *"The uploaded notebook sources do not contain sufficient information to answer this question. Please upload relevant source documents to ground the answer."*
4. **Sub-200ms Retrieval Latency**:
   - In-memory **FAISS** vector search fused with **BM25Okapi** keyword search via Reciprocal Rank Fusion (RRF).
   - Real-time latency telemetry measuring STT, retrieval, LLM Time-to-First-Token (TTFT), and total end-to-end execution.
5. **Multilingual Capabilities (English & Hindi)**:
   - Uses `paraphrase-multilingual-MiniLM-L12-v2` dense embeddings, enabling cross-lingual queries (e.g., asking in Hindi and retrieving relevant passages from English documents).
6. **Responsive Dual-Theme Web Interface**:
   - Clean, professional design with zero emojis, celebratory confetti bursts, mobile haptics, and toast notifications.
   - Dynamic **Dark Mode** (Pookie Pink & Obsidian Black) and **Light Mode** (Matcha Green, Crisp White, and Slate).

---

## Quickstart (Local)

### 1. Launch Interactive Web App
- **One-Click (Windows)**: Double-click `run.bat` in the project folder.
- **Via PowerShell**:
  ```powershell
  .\.venv\Scripts\python.exe app.py
  ```
Open `http://127.0.0.1:7860` in your browser.

### 2. Run Automated Tests & Benchmarks
```powershell
.\.venv\Scripts\python.exe test_pipeline.py
.\.venv\Scripts\python.exe benchmark.py
```
