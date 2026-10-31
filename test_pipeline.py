import unittest
import os
from pathlib import Path
from src.config import Config
from src.pipeline import VoiceRAGPipeline
from src.rag.msmarco_loader import MSMarcoLoader

class TestVoiceRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = VoiceRAGPipeline()
        cls.pipeline.initialize(force_reindex=True)

    def test_01_dataset_loading(self):
        """Verify MS MARCO sample dataset is correctly parsed with IDs."""
        passages = MSMarcoLoader.load_tsv(Config.SAMPLE_MSMARCO_PATH)
        self.assertGreater(len(passages), 5)
        pids = [p.passage_id for p in passages]
        self.assertIn("100001", pids)
        self.assertIn("100002", pids)

    def test_02_factual_retrieval(self):
        """Verify accurate hybrid retrieval for a known question in dataset."""
        answer, audio, profile, passages = self.pipeline.run(
            text_query="What is the speed of light?",
            synthesize_voice=False
        )
        self.assertTrue(profile.passed_guardrail)
        self.assertGreater(len(passages), 0)
        top_pid = passages[0]["passage_id"]
        self.assertEqual(top_pid, "100001")
        # Check latency is populated
        self.assertGreater(profile.retrieval_ms, 0)
        print(f"\n[Test Factual] Latency: {profile.retrieval_ms:.2f}ms | Top Passage: {top_pid}")

    def test_03_anti_hallucination_refusal(self):
        """Verify strict refusal when asking out-of-domain questions."""
        trick_query = "What was the exact winning score of the Martian championship final in 2099?"
        answer, audio, profile, passages = self.pipeline.run(
            text_query=trick_query,
            synthesize_voice=False
        )
        # Should be blocked by guardrail or strictly refused
        self.assertTrue(
            profile.refusal_triggered or Config.STRICT_REFUSAL_MESSAGE in answer
        )
        print(f"\n[Test Refusal] Correctly Refused: {answer.strip()[:60]}...")

    def test_04_multilingual_cross_lingual_retrieval(self):
        """Verify asking in Hindi retrieves corresponding English MS MARCO passage."""
        hindi_query = "माइटोकॉन्ड्रिया का कार्य क्या है?"  # "What is the function of mitochondria?"
        answer, audio, profile, passages = self.pipeline.run(
            text_query=hindi_query,
            synthesize_voice=False
        )
        self.assertTrue(profile.passed_guardrail)
        self.assertGreater(len(passages), 0)
        # Mitochondria passage is 100002
        retrieved_pids = [p["passage_id"] for p in passages]
        self.assertIn("100002", retrieved_pids[:2])
        print(f"\n[Test Multilingual] Hindi query retrieved PID: {retrieved_pids[0]}")

    def test_05_tts_synthesis(self):
        """Verify Text-to-Speech produces a valid audio file."""
        from src.audio.tts import TextToSpeech
        tts = TextToSpeech()
        audio_file, latency = tts.synthesize("Testing voice synthesis system.", language="en")
        self.assertTrue(Path(audio_file).exists())
        self.assertGreater(os.path.getsize(audio_file), 100)
        print(f"\n[Test TTS] Synthesized in {latency:.1f}ms: {audio_file}")

if __name__ == "__main__":
    unittest.main()
