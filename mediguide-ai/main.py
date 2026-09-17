"""
main.py
--------
Command-line interface for MediGuide AI.
Run with:  python main.py
"""

from agent import MediGuideAgent


def print_header():
    print("=" * 60)
    print(" MediGuide AI - Health Symptom Advisor (EDUCATIONAL DEMO)")
    print("=" * 60)
    print(MediGuideAgent.DISCLAIMER)
    print("-" * 60)


def print_menu():
    print("\nWhat would you like to do?")
    print("  1. Enter symptoms -> get rule-based possible conditions")
    print("  2. Check probability of a SPECIFIC condition")
    print("  3. Get ML-based risk level (Low/Medium/High)")
    print("  4. Find route to the relevant specialist")
    print("  5. View session log")
    print("  6. Exit")


def get_symptom_list():
    raw = input("Enter symptoms, comma-separated (e.g. fever, cough, fatigue): ")
    return [s.strip() for s in raw.split(",") if s.strip()]


def handle_consult(agent):
    symptoms = get_symptom_list()
    result = agent.consult(symptoms)
    if "error" in result:
        print(f"  -> {result['error']}")
        return

    print("\n[Rule-Based Inference]")
    if result["rule_based_matches"]:
        for m in result["rule_based_matches"]:
            print(f"  {m['condition']:30s} (rule {m['rule_id']}, matched {m['matched_symptoms']})")
    else:
        print("  No rule fully matched these symptoms.")

    print("\n[Probabilistic Ranking - Bayes' theorem]")
    for entry in result["probability_ranking"][:5]:
        print(f"  {entry['condition']:30s} P = {entry['probability']}")


def handle_check_condition(agent):
    kb_conditions = agent.kb.all_conditions()
    print(f"Known conditions: {', '.join(kb_conditions)}")
    condition = input("Which condition do you want to check? ")
    symptoms = get_symptom_list()
    result = agent.check_condition(condition, symptoms)
    bc = result["backward_chaining"]
    print(f"\n[Backward Chaining] Supported by rules: {bc.get('supported')}")
    if not bc.get("supported", True) and "missing_symptoms" in bc:
        print(f"  Missing symptoms to fully match rule {bc['rule_id']}: {bc['missing_symptoms']}")
    print(f"[Bayesian Probability] P({condition} | symptoms) = {result['probability']}")


def handle_predict_risk(agent):
    try:
        age = int(input("Age: "))
    except ValueError:
        print("  -> Invalid age.")
        return
    flags = {}
    for symptom in ["fever", "cough", "fatigue", "breathlessness", "headache", "nausea", "rash"]:
        ans = input(f"{symptom.capitalize()} present? (y/n): ").strip().lower()
        flags[symptom] = 1 if ans == "y" else 0

    result = agent.predict_risk(age, flags)
    print(f"\n[ML Risk Classifier] Predicted risk level: {result['predicted_risk']}")
    print(f"  Class probabilities: {result['class_probabilities']}")


def handle_route(agent):
    kb_conditions = agent.kb.all_conditions()
    print(f"Known conditions: {', '.join(kb_conditions)}")
    condition = input("Condition to route for: ")
    route = agent.route_to_specialist(condition)
    print(f"\n[Search] Recommended specialist: {route['specialist']}")
    print(f"  BFS path (fewest hops):   {route['bfs_path']}  ({route['bfs_hops']} hops)")
    print(f"  Shortest path (by cost):  {route['shortest_path']}  (cost {route['shortest_cost']})")


def handle_view_log(agent):
    log = agent.get_session_log()
    if not log:
        print("  No actions recorded yet this session.")
        return
    for entry in log:
        print(f"  [{entry['time']}] {entry['action']}")


def main():
    print_header()
    agent = MediGuideAgent()

    actions = {
        "1": handle_consult,
        "2": handle_check_condition,
        "3": handle_predict_risk,
        "4": handle_route,
        "5": handle_view_log,
    }

    while True:
        print_menu()
        choice = input("Choice: ").strip()
        if choice == "6":
            print("Goodbye. Stay healthy!")
            break
        action = actions.get(choice)
        if action:
            action(agent)
        else:
            print("  -> Invalid choice, try again.")


if __name__ == "__main__":
    main()
