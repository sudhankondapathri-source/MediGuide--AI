# Problem Statement

## Problem Statement
Students learning core AI concepts (agents, search, knowledge representation,
probability, and machine learning) often practice each topic in isolation
through separate small exercises, which makes it hard to see how these
techniques work together inside one real system. There is a need for a single,
runnable project that demonstrates a believable use case where rule-based
reasoning, probabilistic inference, search, and supervised learning are
combined under one intelligent agent.

## Scope of the Project
MediGuide AI is a command-line educational demo of a health-symptom advisory
agent. Given a set of symptoms, it:
1. Applies propositional-logic rules (forward and backward chaining) to
   suggest possible conditions.
2. Computes a Bayesian probability distribution over conditions.
3. Uses a supervised ML classifier trained on a synthetic dataset to predict
   a general risk level (Low/Medium/High).
4. Uses graph search (BFS and Uniform Cost Search) to route the user to the
   appropriate specialist within a simulated clinic network.

**Explicitly out of scope:** real medical diagnosis, real patient data, and
any clinical decision-making. This is an AI-coursework demonstration only.

## Target Users
- Students and evaluators assessing the application of AI course concepts.
- (Fictional, for demo purposes) a person exploring how their symptoms map
  to general condition categories and which type of clinic specialist they
  might look for.

## High-Level Features
- Rule-based symptom-to-condition inference engine (forward & backward
  chaining) with an externally editable rule base (`data/rules.json`).
- From-scratch Bayesian probability engine for ranked condition likelihoods.
- Random Forest ML classifier for risk-level prediction, trained on a
  synthetic, clearly-labeled dataset.
- Graph-based specialist routing using BFS and Uniform Cost Search.
- Session logging of every consultation for traceability.
- Fully menu-driven CLI, no GUI dependency.
