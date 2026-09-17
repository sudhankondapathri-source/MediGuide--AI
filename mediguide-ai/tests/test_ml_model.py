"""
Run with:  python -m unittest tests/test_ml_model.py
(from the project root, with the venv activated)
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_model import RiskClassifier


class TestRiskClassifier(unittest.TestCase):
    def setUp(self):
        self.clf = RiskClassifier()
        self.clf.train()

    def test_training_reaches_reasonable_accuracy(self):
        # Synthetic data with noise - expect the model to beat random guessing (33%)
        self.assertGreater(self.clf.last_accuracy, 0.5)

    def test_predict_returns_valid_class(self):
        pred, probs = self.clf.predict(age=70, fever=1, cough=1, fatigue=1,
                                        breathlessness=1, headache=0, nausea=0, rash=0)
        self.assertIn(pred, ["Low", "Medium", "High"])
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)

    def test_high_risk_profile_skews_high(self):
        # Elderly patient with breathlessness and many symptoms -> expect High
        # to have a meaningfully higher probability than Low.
        _, probs = self.clf.predict(age=80, fever=1, cough=1, fatigue=1,
                                     breathlessness=1, headache=1, nausea=1, rash=0)
        self.assertGreater(probs.get("High", 0), probs.get("Low", 0))


if __name__ == "__main__":
    unittest.main()
