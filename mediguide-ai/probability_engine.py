"""
probability_engine.py
-----------------------
Probabilistic Reasoning module.

Implements Bayes' theorem FROM SCRATCH (no library shortcuts) to demonstrate
understanding of:
  - Prior probability P(Disease)
  - Conditional probability P(Symptom | Disease)   [a "naive" independence
    assumption is used across symptoms, same simplification Naive Bayes makes]
  - Posterior probability P(Disease | Symptoms) via Bayes' rule, normalized
    across all candidate diseases so the outputs sum to 1.

P(D | S1, S2, ..., Sn)  is proportional to  P(D) * P(S1|D) * P(S2|D) * ... * P(Sn|D)

For a symptom that is ABSENT, we multiply by (1 - P(symptom|D)) instead,
which is standard Naive Bayes handling of binary features.
"""

import json
import os

PROB_TABLE_PATH = os.path.join(os.path.dirname(__file__), "data", "prob_table.json")


class ProbabilityEngine:
    def __init__(self, table_path=PROB_TABLE_PATH):
        with open(table_path, "r") as f:
            data = json.load(f)
        self.diseases = data["diseases"]
        # the full universe of symptoms the table knows about
        self.all_symptoms = set()
        for d in self.diseases.values():
            self.all_symptoms.update(d["symptom_given_disease"].keys())

    def posterior_for_disease(self, disease, present_symptoms):
        """Unnormalized P(disease) * product of P(symptom|disease) terms."""
        info = self.diseases[disease]
        prob = info["prior"]
        present = set(s.lower() for s in present_symptoms)

        for symptom in self.all_symptoms:
            p_given_d = info["symptom_given_disease"].get(symptom, 0.05)  # small default
            if symptom in present:
                prob *= p_given_d
            else:
                prob *= (1 - p_given_d)
        return prob

    def diagnose(self, present_symptoms, top_n=5):
        """
        Returns a normalized probability distribution over all known diseases
        given the presented symptoms, sorted descending, top_n entries.
        """
        raw_scores = {d: self.posterior_for_disease(d, present_symptoms) for d in self.diseases}
        total = sum(raw_scores.values())
        if total == 0:
            return []

        normalized = {d: score / total for d, score in raw_scores.items()}
        ranked = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        return [{"condition": d, "probability": round(p, 4)} for d, p in ranked[:top_n]]

    def probability_of_specific_condition(self, condition, present_symptoms):
        """Convenience wrapper: normalized probability of just ONE named condition."""
        full = self.diagnose(present_symptoms, top_n=len(self.diseases))
        for entry in full:
            if entry["condition"].lower() == condition.lower():
                return entry["probability"]
        return 0.0


if __name__ == "__main__":
    engine = ProbabilityEngine()
    symptoms = ["fever", "cough", "fatigue"]
    print(f"Diagnosis distribution for symptoms {symptoms}:")
    for entry in engine.diagnose(symptoms):
        print(f"  {entry['condition']:30s} P = {entry['probability']}")
