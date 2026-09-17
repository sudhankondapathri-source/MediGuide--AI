"""
generate_dataset.py
--------------------
Generates a SYNTHETIC (not real patient data) dataset for training the
ML risk classifier used in ml_model.py.

Risk logic baked in on purpose (so the model has real signal to learn):
  - Higher age -> higher risk
  - More symptoms present -> higher risk
  - Breathlessness present -> pushes risk up strongly
  - Random noise added so classes aren't perfectly separable (realistic)

Run once to (re)create data/training_data.csv:
    python data/generate_dataset.py
"""

import csv
import random

random.seed(42)

FIELDNAMES = [
    "age", "fever", "cough", "fatigue", "breathlessness",
    "headache", "nausea", "rash", "num_symptoms", "risk_level"
]


def generate_row():
    age = random.randint(5, 85)
    fever = random.choice([0, 1])
    cough = random.choice([0, 1])
    fatigue = random.choice([0, 1])
    breathlessness = random.choice([0, 1]) if random.random() < 0.3 else 0
    headache = random.choice([0, 1])
    nausea = random.choice([0, 1]) if random.random() < 0.3 else 0
    rash = random.choice([0, 1]) if random.random() < 0.15 else 0

    num_symptoms = sum([fever, cough, fatigue, breathlessness, headache, nausea, rash])

    score = 0.0
    score += (age / 85) * 2.0
    score += num_symptoms * 0.8
    score += breathlessness * 2.5
    score += random.uniform(-0.7, 0.7)  # noise

    if score < 2.0:
        risk = "Low"
    elif score < 4.0:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "age": age, "fever": fever, "cough": cough, "fatigue": fatigue,
        "breathlessness": breathlessness, "headache": headache,
        "nausea": nausea, "rash": rash, "num_symptoms": num_symptoms,
        "risk_level": risk
    }


def main(n_rows=400, out_path="data/training_data.csv"):
    rows = [generate_row() for _ in range(n_rows)]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {n_rows} synthetic rows to {out_path}")


if __name__ == "__main__":
    main()
