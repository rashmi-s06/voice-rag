import os
import gradio as gr
from pathlib import Path
from src.config import Config
from src.pipeline import VoiceRAGPipeline

# Initialize the pipeline globally
pipeline = VoiceRAGPipeline()
pipeline.initialize()

def process_interaction(audio_file, text_input, api_key_input, voice_enabled):
    """Processes user voice or text input and generates grounded answer + telemetry."""
    # Update API key if provided by user in sidebar
    if api_key_input and api_key_input.strip():
        pipeline.llm.set_api_key(api_key_input.strip())

    audio_path = audio_file if audio_file else None
    query_text = text_input.strip() if text_input else None

    if not audio_path and not query_text:
        return (
            "Please speak into the microphone, upload an audio file, or type a question.",
            None,
            "No query submitted.",
            ""
        )

    # Execute pipeline
    answer, audio_out, profile, passages = pipeline.run(
        audio_path=audio_path,
        text_query=query_text,
        synthesize_voice=voice_enabled
    )

    # Format telemetry breakdown
    telemetry_md = profile.summary_markdown()

    # Format retrieved passages inspector
    passages_md_list = []
    if passages:
        for idx, p in enumerate(passages, 1):
            pid = p.get("passage_id", "N/A")
            rrf = p.get("rrf_score", 0.0)
            dense = p.get("dense_score", 0.0)
            bm25 = p.get("bm25_score", 0.0)
            text = p.get("text", "")
            passages_md_list.append(
                f"**#{idx} [Passage ID: {pid}]** (RRF: `{rrf:.4f}`, Cosine: `{dense:.3f}`, BM25: `{bm25:.2f}`)\n> {text}\n"
            )
        passages_inspector = "\n\n".join(passages_md_list)
    else:
        passages_inspector = "*No passages met the relevance confidence threshold. Refusal guardrail triggered to prevent hallucination.*"

    return answer, audio_out, telemetry_md, passages_inspector

# Gradio Dashboard Theme and Interface
custom_css = """
.benchmark-card { border-radius: 8px; padding: 12px; margin-bottom: 10px; }
.metric-box { font-weight: bold; }
"""

with gr.Blocks(title="Voice-Enabled RAG System (MS MARCO)", css=custom_css) as demo:
    gr.Markdown("""
    # 🎙️ Voice-Enabled RAG System
    ### Strictly Grounded on MS MARCO Dataset | Sub-200ms Latency Target | Multilingual (English & Hindi)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ System Settings")
            api_key_box = gr.Textbox(
                label="Gemini API Key",
                placeholder="Enter Gemini API key (or set GEMINI_API_KEY env)",
                type="password",
                value=Config.GEMINI_API_KEY
            )
            voice_toggle = gr.Checkbox(label="Enable Voice Response (TTS)", value=True)
            gr.Markdown("""
            **Guardrails Active:**
            - 🛡️ Zero-Hallucination Policy
            - 🎯 MS MARCO Strict Passage Citations
            - 🌐 Multilingual STT & Cross-Lingual Retrieval
            - ⚡ FAISS Vector Index + BM25 Hybrid Fusion
            """)

        with gr.Column(scale=2):
            gr.Markdown("### 🗣️ Spoken Voice Input")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record Query (Speak in English or Hindi)"
            )
            text_input = gr.Textbox(
                label="Or Type Query (Fallback)",
                placeholder="e.g. 'What is the speed of light?' or 'माइटोकॉन्ड्रिया क्या है?'"
            )
            submit_btn = gr.Button("🚀 Run Voice RAG Pipeline", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Grounded Response")
            output_answer = gr.Markdown(label="Factual Answer")
            output_audio = gr.Audio(label="Spoken Voice Output", autoplay=True)

        with gr.Column(scale=1):
            gr.Markdown("### ⏱️ Latency Telemetry Breakdown")
            output_telemetry = gr.Markdown(value="*Latency timings will appear here after query execution.*")

    with gr.Accordion("🔍 Retrieved MS MARCO Passages Inspector (Anti-Hallucination Evidence)", open=False):
        output_passages = gr.Markdown(value="*Inspect the exact passages retrieved from the dataset.*")

    submit_btn.click(
        fn=process_interaction,
        inputs=[audio_input, text_input, api_key_box, voice_toggle],
        outputs=[output_answer, output_audio, output_telemetry, output_passages]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
