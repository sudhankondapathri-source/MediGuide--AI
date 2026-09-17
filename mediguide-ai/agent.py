"""
agent.py
---------
Intelligent Agent core.

PEAS description (Module 1 - Intelligent Agents & Environments):
  Performance measure : Relevance and safety of the advice given, correct
                         reasoning trace, responsiveness.
  Environment          : A single-user interactive text session (partially
                          observable - the agent only knows what the user
                          types; episodic in the sense each symptom query
                          stands alone, though a session log is kept).
  Actuators            : Text output to the terminal (possible conditions,
                          probabilities, risk level, specialist route).
  Sensors              : Text input typed by the user (symptoms, choices).

This agent is a SIMPLE REFLEX + MODEL-BASED hybrid: it doesn't just react
to raw input, it maintains a small internal model (the session log) and
combines THREE separate reasoning strategies (rule-based, probabilistic,
learned) before acting - which is why MediGuideAgent.consult() calls all
three and merges their outputs rather than picking just one.
"""

import logging
from datetime import datetime

from knowledge_base import KnowledgeBase
from probability_engine import ProbabilityEngine
from ml_model import RiskClassifier
from search_algorithms import ClinicGraph

logging.basicConfig(
    filename="mediguide_session.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


class MediGuideAgent:
    DISCLAIMER = (
        "MediGuide AI is an EDUCATIONAL AI-course project demonstrating rule-based, "
        "probabilistic, and ML reasoning. It is NOT a medical device and must never "
        "be used for real diagnosis or treatment decisions."
    )

    def __init__(self):
        self.kb = KnowledgeBase()
        self.prob_engine = ProbabilityEngine()
        self.graph = ClinicGraph()
        self.classifier = RiskClassifier()
        self.classifier.train()  # train once at startup
        self.session_log = []
        logging.info("Agent initialized. %s", self.DISCLAIMER)

    def _record(self, action, details):
        entry = {"time": datetime.now().isoformat(timespec="seconds"),
                  "action": action, "details": details}
        self.session_log.append(entry)
        logging.info("%s | %s", action, details)

    def consult(self, symptoms):
        """
        Full consultation: runs rule-based inference AND probabilistic
        diagnosis over the same symptom list, then reports both so the
        user can compare a logic-based view with a probability-based view.
        """
        symptoms = [s.strip().lower() for s in symptoms if s.strip()]
        if not symptoms:
            return {"error": "No valid symptoms provided."}

        rule_matches = self.kb.forward_chain(symptoms)
        prob_ranking = self.prob_engine.diagnose(symptoms)

        result = {
            "symptoms": symptoms,
            "rule_based_matches": rule_matches,
            "probability_ranking": prob_ranking,
        }
        self._record("consult", result)
        return result

    def check_condition(self, condition, symptoms):
        """Backward-chaining goal check + exact Bayesian probability for one named condition."""
        symptoms = [s.strip().lower() for s in symptoms if s.strip()]
        bc_result = self.kb.backward_chain(condition, symptoms)
        probability = self.prob_engine.probability_of_specific_condition(condition, symptoms)
        result = {"condition": condition, "backward_chaining": bc_result, "probability": probability}
        self._record("check_condition", result)
        return result

    def predict_risk(self, age, symptom_flags):
        """symptom_flags: dict with keys fever/cough/fatigue/breathlessness/headache/nausea/rash -> 0/1"""
        pred, probs = self.classifier.predict(age=age, **symptom_flags)
        result = {"age": age, "symptom_flags": symptom_flags,
                   "predicted_risk": pred,
                   "class_probabilities": {k: round(v, 3) for k, v in probs.items()}}
        self._record("predict_risk", result)
        return result

    def route_to_specialist(self, condition):
        route = self.graph.find_specialist_route(condition)
        self._record("route_to_specialist", route)
        return route

    def get_session_log(self):
        return self.session_log
