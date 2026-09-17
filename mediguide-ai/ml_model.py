"""
ml_model.py
------------
Supervised Machine Learning module.

Trains a classifier on data/training_data.csv to predict a patient's
risk_level (Low / Medium / High) from age + which symptoms are present.

Uses scikit-learn's RandomForestClassifier (an ensemble of decision trees) -
chosen because it handles the mixed numeric/binary features here well and
gives a feature_importances_ attribute we can report on for interpretability.

Model selection rationale (for the report):
  - Dataset is small (a few hundred rows) and tabular -> tree-based models
    are a strong, fast, easy-to-explain baseline (no need for deep learning).
  - RandomForest > a single Decision Tree here because it reduces overfitting
    via averaging many trees trained on bootstrapped samples.
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "training_data.csv")

FEATURE_COLUMNS = [
    "age", "fever", "cough", "fatigue", "breathlessness",
    "headache", "nausea", "rash", "num_symptoms"
]
LABEL_COLUMN = "risk_level"


class RiskClassifier:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.trained = False
        self.last_accuracy = None

    def train(self, test_size=0.2, verbose=False):
        df = pd.read_csv(self.data_path)
        X = df[FEATURE_COLUMNS]
        y = df[LABEL_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        self.last_accuracy = accuracy_score(y_test, preds)
        self.trained = True

        if verbose:
            print(f"Test accuracy: {self.last_accuracy:.2%}")
            print(classification_report(y_test, preds))

        return self.last_accuracy

    def predict(self, age, fever, cough, fatigue, breathlessness, headache, nausea, rash):
        if not self.trained:
            self.train()
        num_symptoms = sum([fever, cough, fatigue, breathlessness, headache, nausea, rash])
        row = pd.DataFrame([{
            "age": age, "fever": fever, "cough": cough, "fatigue": fatigue,
            "breathlessness": breathlessness, "headache": headache,
            "nausea": nausea, "rash": rash, "num_symptoms": num_symptoms
        }])
        prediction = self.model.predict(row)[0]
        probabilities = dict(zip(self.model.classes_, self.model.predict_proba(row)[0]))
        return prediction, probabilities

    def feature_importance_report(self):
        if not self.trained:
            self.train()
        importances = dict(zip(FEATURE_COLUMNS, self.model.feature_importances_))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    clf = RiskClassifier()
    clf.train(verbose=True)

    print("\nSample prediction (age=70, fever+cough+breathlessness present):")
    pred, probs = clf.predict(age=70, fever=1, cough=1, fatigue=0,
                               breathlessness=1, headache=0, nausea=0, rash=0)
    print(f"  Predicted risk: {pred}")
    print(f"  Class probabilities: { {k: round(v, 3) for k, v in probs.items()} }")

    print("\nFeature importances:")
    for feat, score in clf.feature_importance_report().items():
        print(f"  {feat:15s} {score:.3f}")
