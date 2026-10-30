import unittest
import os
import tempfile
from pathlib import Path
from src.config import Config
from src.pipeline import VoiceRAGPipeline

class TestNotebookLMPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = VoiceRAGPipeline()
        cls.pipeline.initialize(load_starter_demo=False)

    def test_01_empty_notebook_guardrail(self):
        """Verify querying an empty notebook returns the empty state prompt and blocks hallucination."""
        self.assertEqual(self.pipeline.get_total_passages_count(), 0)
        answer, audio, profile, passages = self.pipeline.run(
            text_query="What is quantum mechanics?",
            synthesize_voice=False
        )
        self.assertTrue(profile.refusal_triggered)
        self.assertIn("notebook is empty", answer.lower())
        self.assertEqual(len(passages), 0)
        print("\n[Test Empty Notebook] Correctly blocked query on empty notebook.")

    def test_02_add_file_source(self):
        """Verify adding a user document file (.txt) creates a source with proper chunk IDs."""
        temp_file = Path(tempfile.gettempdir()) / "quantum_physics.txt"
        temp_file.write_text(
            "Quantum computing utilizes quantum bits or qubits to perform calculations at unprecedented speeds.\n\n"
            "Superposition allows qubits to exist in multiple states simultaneously.",
            encoding="utf-8"
        )
        source, total_chunks = self.pipeline.add_source_file(str(temp_file))
        self.assertEqual(source.name, "quantum_physics.txt")
        self.assertEqual(source.num_chunks, 2)
        self.assertEqual(total_chunks, 2)
        self.assertIn("quantum_physics.txt#Chunk1", [p.passage_id for p in source.passages])
        print(f"\n[Test Add Source] Successfully added source with {total_chunks} chunks.")

    def test_03_factual_source_retrieval(self):
        """Verify accurate retrieval against user-uploaded source document."""
        answer, audio, profile, passages = self.pipeline.run(
            text_query="What do quantum computers utilize?",
            synthesize_voice=False
        )
        self.assertTrue(profile.passed_guardrail)
        self.assertGreater(len(passages), 0)
        top_passage = passages[0]
        self.assertIn("qubits", top_passage["text"])
        self.assertIn("quantum_physics.txt", top_passage["passage_id"])
        print(f"\n[Test Factual Retrieval] Matched passage ID: {top_passage['passage_id']} in {profile.retrieval_ms:.1f}ms")

    def test_04_anti_hallucination_refusal(self):
        """Verify strict refusal when asking about something outside user sources."""
        trick_query = "What was the exact winning score of the Martian championship final in 2099?"
        answer, audio, profile, passages = self.pipeline.run(
            text_query=trick_query,
            synthesize_voice=False
        )
        self.assertTrue(profile.refusal_triggered)
        self.assertIn(Config.STRICT_REFUSAL_MESSAGE, answer)
        print(f"\n[Test Anti-Hallucination] Correctly refused out-of-domain query: {answer[:60]}...")

    def test_05_multilingual_source_retrieval(self):
        """Verify Hindi spoken/text queries retrieve from English user documents."""
        hindi_query = "क्वांटम कंप्यूटर क्या उपयोग करते हैं?"  # "What do quantum computers use?"
        answer, audio, profile, passages = self.pipeline.run(
            text_query=hindi_query,
            synthesize_voice=False
        )
        self.assertTrue(profile.passed_guardrail)
        self.assertGreater(len(passages), 0)
        self.assertIn("quantum_physics.txt", passages[0]["passage_id"])
        print(f"\n[Test Multilingual] Hindi query retrieved from user source: {passages[0]['passage_id']}")

    def test_06_source_removal(self):
        """Verify deleting a source document immediately removes its passages from the index."""
        sources = self.pipeline.get_sources_summary()
        self.assertGreater(len(sources), 0)
        first_src_id = sources[0]["source_id"]
        remaining_chunks = self.pipeline.remove_source(first_src_id)
        self.assertEqual(remaining_chunks, 0)
        self.assertEqual(self.pipeline.get_total_passages_count(), 0)
        print("\n[Test Remove Source] Source cleanly removed and index cleared.")

    def test_07_starter_pack_and_voice_loop(self):
        """Verify loading starter demo pack and running full voice loop (STT -> RAG -> TTS)."""
        source, count = self.pipeline.load_starter_pack()
        self.assertGreater(count, 0)
        self.assertEqual(source.name, "Starter_Knowledge_Base.txt")

        from src.audio.tts import TextToSpeech
        tts = TextToSpeech()
        audio_in, _ = tts.synthesize("What is the powerhouse of the cell?", language="en")

        answer, audio_out, profile, passages = self.pipeline.run(
            audio_path=audio_in,
            synthesize_voice=True
        )
        self.assertGreater(profile.stt_ms, 0)
        self.assertTrue(profile.passed_guardrail)
        self.assertGreater(len(passages), 0)
        self.assertIn("mitochondria", passages[0]["text"].lower())
        self.assertIsNotNone(audio_out)
        self.assertTrue(Path(audio_out).exists())
        print(f"\n[Test Voice Loop] STT: {profile.stt_ms:.1f}ms | Retrieval: {profile.retrieval_ms:.1f}ms | TTS: {profile.tts_ms:.1f}ms")

if __name__ == "__main__":
    unittest.main()
