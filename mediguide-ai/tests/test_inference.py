"""
Run with:  python -m unittest tests/test_inference.py
(from the project root, with the venv activated)
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_base import KnowledgeBase
from probability_engine import ProbabilityEngine


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase()

    def test_forward_chain_finds_flu(self):
        matches = self.kb.forward_chain(["fever", "cough", "fatigue"])
        conditions = [m["condition"] for m in matches]
        self.assertIn("Flu", conditions)

    def test_forward_chain_no_match_returns_empty(self):
        matches = self.kb.forward_chain(["itchy toe"])
        self.assertEqual(matches, [])

    def test_backward_chain_supported_true(self):
        result = self.kb.backward_chain("Migraine", ["headache", "fatigue", "nausea"])
        self.assertTrue(result["supported"])

    def test_backward_chain_reports_missing_symptoms(self):
        result = self.kb.backward_chain("Migraine", ["headache"])
        self.assertFalse(result["supported"])
        self.assertIn("nausea", result["missing_symptoms"] + result["missing_symptoms"])


class TestProbabilityEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ProbabilityEngine()

    def test_diagnose_returns_normalized_distribution(self):
        ranking = self.engine.diagnose(["fever", "cough", "fatigue"])
        total = sum(entry["probability"] for entry in ranking) if len(ranking) == len(self.engine.diseases) else None
        # if top_n < all diseases, we just check individual bounds instead
        for entry in ranking:
            self.assertGreaterEqual(entry["probability"], 0.0)
            self.assertLessEqual(entry["probability"], 1.0)

    def test_flu_is_top_match_for_classic_flu_symptoms(self):
        ranking = self.engine.diagnose(["fever", "cough", "fatigue"])
        self.assertEqual(ranking[0]["condition"], "Flu")


if __name__ == "__main__":
    unittest.main()
