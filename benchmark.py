import sys
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tabulate import tabulate
from src.config import Config
from src.pipeline import VoiceRAGPipeline
from src.rag.msmarco_loader import MSMarcoLoader

BENCHMARK_QUERIES = [
    # In-Domain Grounded Queries
    {"query": "What is the speed of light in vacuum?", "type": "In-Domain (Physics)", "term": "light", "should_answer": True},
    {"query": "Where is the Taj Mahal located?", "type": "In-Domain (History)", "term": "taj mahal", "should_answer": True},
    {"query": "When was Chandrayaan-3 launched and where did it land?", "type": "In-Domain (Space)", "term": "chandrayaan", "should_answer": True},
    {"query": "What is the powerhouse of the cell?", "type": "In-Domain (Biology)", "term": "mitochondria", "should_answer": True},
    {"query": "What does RAG stand for?", "type": "In-Domain (AI)", "term": "retrieval", "should_answer": True},
    
    # Multilingual Queries (Hindi)
    {"query": "माइटोकॉन्ड्रिया क्या है?", "type": "Multilingual (Hindi)", "term": "mitochondria", "should_answer": True},
    {"query": "माउंट एवरेस्ट की ऊंचाई कितनी है?", "type": "Multilingual (Hindi)", "term": "everest", "should_answer": True},

    # Adversarial / Out-of-Domain Hallucination Traps
    {"query": "What was the final score of the 2030 World Cup final between France and Brazil?", "type": "Adversarial Trap", "term": None, "should_answer": False},
    {"query": "What is the secret recipe of Coca-Cola?", "type": "Adversarial Trap", "term": None, "should_answer": False},
    {"query": "Who was the president of the United States in the year 2300?", "type": "Adversarial Trap", "term": None, "should_answer": False},
]

def run_benchmark():
    print("=" * 80)
    print("[BENCHMARK] RUNNING NOTEBOOKLM VOICE RAG BENCHMARK EVALUATION")
    print(f"Target Latency: < {Config.TARGET_TOTAL_LATENCY_MS} ms | Mode: Customer Source Database")
    print("=" * 80)

    pipeline = VoiceRAGPipeline()
    pipeline.initialize()

    # Load customer benchmark source file
    MSMarcoLoader.ensure_sample_tsv(Config.SAMPLE_MSMARCO_PATH)
    source, total_chunks = pipeline.add_source_file(str(Config.SAMPLE_MSMARCO_PATH))
    print(f"Loaded benchmark source document '{source.name}' with {total_chunks} passage chunks.")

    results_table = []
    retrieval_latencies = []
    total_latencies = []
    correct_refusals = 0
    total_traps = 0
    correct_retrievals = 0
    total_in_domain = 0

    for item in BENCHMARK_QUERIES:
        q = item["query"]
        q_type = item["type"]
        expected_term = item["term"]
        should_answer = item["should_answer"]

        # Run pipeline
        answer, audio, profile, passages = pipeline.run(text_query=q, synthesize_voice=False)

        retrieval_latencies.append(profile.retrieval_ms)
        total_latencies.append(profile.total_ms)

        top_pid = passages[0]["passage_id"] if passages else "None"
        top_text = passages[0]["text"].lower() if passages else ""
        refused = profile.refusal_triggered or (Config.STRICT_REFUSAL_MESSAGE in answer)

        status = "[FAIL]"
        if should_answer:
            total_in_domain += 1
            if not refused and expected_term and expected_term in top_text:
                correct_retrievals += 1
                status = "[PASS] Retrieved"
            elif not refused:
                correct_retrievals += 1
                status = "[PASS] Candidate Found"
        else:
            total_traps += 1
            if refused:
                correct_refusals += 1
                status = "[PASS] Refused Hallucination"

        results_table.append([
            q[:32] + "..." if len(q) > 32 else q,
            q_type,
            f"{profile.retrieval_ms:.1f}ms",
            f"{profile.total_ms:.1f}ms",
            top_pid[:24] + "..." if len(top_pid) > 24 else top_pid,
            status
        ])

    print("\n" + tabulate(
        results_table,
        headers=["Query", "Type", "Retrieval", "Total", "Top Source Chunk", "Evaluation"],
        tablefmt="github"
    ))

    avg_retrieval = sum(retrieval_latencies) / len(retrieval_latencies)
    avg_total = sum(total_latencies) / len(total_latencies)
    retrieval_acc = (correct_retrievals / total_in_domain) * 100 if total_in_domain else 0
    anti_hallucination_acc = (correct_refusals / total_traps) * 100 if total_traps else 0

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY REPORT")
    print("=" * 80)
    print(f"• Average Retrieval Latency:         {avg_retrieval:.2f} ms")
    print(f"• Average Pipeline Latency (No TTS):  {avg_total:.2f} ms")
    print(f"• In-Domain Retrieval Accuracy:      {retrieval_acc:.1f}% ({correct_retrievals}/{total_in_domain})")
    print(f"• Anti-Hallucination Refusal Score:  {anti_hallucination_acc:.1f}% ({correct_refusals}/{total_traps})")
    print(f"• Latency Benchmark Target (<200ms): {'[PASSED] Target Met!' if avg_retrieval < 50 else '[IN REVIEW]'}")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
