# MediGuide--AI
CLI-based AI health symptom advisor combining rule-based inference, Bayesian reasoning, search algorithms, and a supervised ML risk classifier. Educational AI-course project.
 Overview

MediGuide AI is a command-line intelligent agent that demonstrates four core areas of Artificial Intelligence in one integrated system:

Knowledge Representation & Inference — propositional-logic rules with forward and backward chaining (knowledge_base.py)
Probabilistic Reasoning — a from-scratch Bayes' theorem engine (probability_engine.py)
Search Strategies — BFS and Uniform Cost Search over a clinic network graph (search_algorithms.py)
Supervised Machine Learning — a Random Forest classifier predicting a risk level from symptoms (ml_model.py)

All four are orchestrated by a single intelligent agent (agent.py) with an explicit PEAS description, and exposed through a menu-driven CLI (main.py).

Features
Enter symptoms and get both a rule-based diagnosis and a Bayesian probability ranking over possible conditions.
Ask "could this be condition X?" and get a backward-chaining explanation of exactly which symptoms are missing to confirm it.
Get an ML-predicted risk level (Low / Medium / High) with class probabilities.
Get routed to the correct type of specialist via graph search, with both the fewest-hops path and the true lowest-cost path shown.
Every action in a session is logged to mediguide_session.log.
Technologies / Tools Used
Python 3.10+
pandas — data handling for the ML pipeline
scikit-learn — Random Forest classifier, train/test split, evaluation metrics
Standard library only for the rule engine, Bayes engine, and search algorithms (json, heapq, collections, unittest, logging)
Project Structure
mediguide-ai/
├── agent.py                  # Intelligent agent core (PEAS, orchestration)
├── knowledge_base.py         # Rule-based inference (forward/backward chaining)
├── probability_engine.py     # Bayes' theorem reasoning engine
├── ml_model.py                # Random Forest risk classifier
├── search_algorithms.py      # BFS / Uniform Cost Search over clinic graph
├── main.py                   # CLI entry point
├── requirements.txt
├── statement.md               # Problem statement, scope, target users
├── data/
│   ├── rules.json             # Editable symptom -> condition rules
│   ├── prob_table.json        # Prior + conditional probability table
│   ├── specialist_graph.json  # Clinic network graph + condition mapping
│   ├── generate_dataset.py    # Synthetic dataset generator
│   └── training_data.csv      # Generated synthetic training data
└── tests/
    ├── test_inference.py      # Unit tests for KB + Bayes engine
    └── test_ml_model.py       # Unit tests for the ML classifier
Setup & Installation
1. Prerequisites
Python 3.10 or later installed (python.org)
Git installed (git-scm.com)

Verify:

bash
python --version
git --version
2. Clone the repository
bash
git clone https://github.com/{your-username}/{your-repo-name}.git
cd {your-repo-name}
3. Create and activate a virtual environment
bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
4. Install dependencies
bash
pip install -r requirements.txt
5. (Re)generate the synthetic dataset

A dataset is already included, but you can regenerate it deterministically:

bash
python data/generate_dataset.py
Running the Project

From the project root, with the virtual environment activated:

bash
python main.py

You will see a menu. Enter the number of the option you want and follow the prompts. Example symptom input: fever, cough, fatigue.

Running the Tests
bash
python -m unittest tests.test_inference tests.test_ml_model -v

All 9 unit tests should pass. They check:

Rule-based forward/backward chaining correctness
Bayesian probability outputs are valid distributions
The ML classifier trains above baseline accuracy and produces valid, sensible predictions
Design Notes
Why rules AND probability? Rules give an explainable, deterministic view ("these exact symptoms match this exact rule"), while the Bayes engine gives a graded, more realistic likelihood — the CLI intentionally shows both so their difference is itself part of the demonstration.
Why Random Forest for the ML module? The dataset is small, tabular, and has both numeric and binary features — a tree ensemble is fast, robust to this feature mix, and gives interpretable feature importances without needing a GPU or large data volumes.
Why is the dataset synthetic? No real patient data is used or required; data/generate_dataset.py documents the exact rule used to label synthetic rows so the whole pipeline is transparent and reproducible.
Future Enhancements
Add an A* heuristic (e.g. estimated distance) to the specialist routing search for larger clinic networks.
Persist the trained ML model with joblib instead of retraining on every run.
Move the rule base and probability table into SQLite for a proper database/schema design.
Disclaimer

This project was built to satisfy a university AI-course assignment. It is not reviewed by any medical professional and must not be relied upon for real health decisions.
