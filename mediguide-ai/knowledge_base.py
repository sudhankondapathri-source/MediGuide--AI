"""
knowledge_base.py
------------------
Knowledge Representation & Inference module.

Demonstrates:
  - Propositional-logic-style rules: IF (symptom AND symptom ...) THEN condition
  - Forward chaining: given facts (symptoms), derive all conclusions (conditions)
  - Backward chaining: given a goal (a specific condition), check if the facts support it

Rules are stored externally in data/rules.json so the knowledge base can be
extended without touching code (a maintainability requirement).
"""

import json
import os

RULES_PATH = os.path.join(os.path.dirname(__file__), "data", "rules.json")


class Rule:
    """A single propositional rule: a conjunction of symptoms implying a condition."""

    def __init__(self, rule_id, if_symptoms, then_condition, confidence_note=""):
        self.id = rule_id
        self.if_symptoms = set(s.lower() for s in if_symptoms)
        self.then_condition = then_condition
        self.confidence_note = confidence_note

    def is_satisfied_by(self, facts):
        """facts: a set of known symptom strings (lowercase)."""
        return self.if_symptoms.issubset(facts)

    def __repr__(self):
        return f"<Rule {self.id}: IF {sorted(self.if_symptoms)} THEN {self.then_condition}>"


class KnowledgeBase:
    def __init__(self, rules_path=RULES_PATH):
        self.rules = self._load_rules(rules_path)

    @staticmethod
    def _load_rules(path):
        with open(path, "r") as f:
            data = json.load(f)
        return [
            Rule(r["id"], r["if_symptoms"], r["then_condition"], r.get("confidence_note", ""))
            for r in data["rules"]
        ]

    def forward_chain(self, symptoms):
        """
        Data-driven inference: given a set of symptoms (facts), return every
        condition whose rule is fully satisfied, most-specific (most symptoms
        matched) first.
        """
        facts = set(s.strip().lower() for s in symptoms)
        matches = [r for r in self.rules if r.is_satisfied_by(facts)]
        matches.sort(key=lambda r: len(r.if_symptoms), reverse=True)
        return [
            {
                "rule_id": r.id,
                "condition": r.then_condition,
                "matched_symptoms": sorted(r.if_symptoms),
                "note": r.confidence_note,
            }
            for r in matches
        ]

    def backward_chain(self, goal_condition, symptoms):
        """
        Goal-driven inference: check whether the given symptoms are enough to
        support a SPECIFIC condition the user is asking about (e.g. "could this
        be Migraine?"). Returns a dict explaining which symptoms are missing.
        """
        facts = set(s.strip().lower() for s in symptoms)
        candidate_rules = [r for r in self.rules if r.then_condition.lower() == goal_condition.lower()]

        if not candidate_rules:
            return {"supported": False, "reason": f"No rule exists for condition '{goal_condition}'."}

        best = None
        for r in candidate_rules:
            missing = r.if_symptoms - facts
            if best is None or len(missing) < len(best["missing"]):
                best = {
                    "rule_id": r.id,
                    "required_symptoms": sorted(r.if_symptoms),
                    "missing": sorted(missing),
                }

        supported = len(best["missing"]) == 0
        return {
            "supported": supported,
            "rule_id": best["rule_id"],
            "required_symptoms": best["required_symptoms"],
            "missing_symptoms": best["missing"],
        }

    def all_conditions(self):
        return sorted(set(r.then_condition for r in self.rules))


if __name__ == "__main__":
    # Quick standalone self-test
    kb = KnowledgeBase()
    print("Loaded rules:", len(kb.rules))

    print("\nForward chaining test (fever, cough, fatigue):")
    for match in kb.forward_chain(["fever", "cough", "fatigue"]):
        print(" ", match)

    print("\nBackward chaining test (goal=Migraine, symptoms=[headache, fatigue]):")
    print(" ", kb.backward_chain("Migraine", ["headache", "fatigue"]))
