"""
search_algorithms.py
----------------------
Search Strategies module (uninformed + informed).

Applied here to a small "clinic network" graph: given a diagnosed condition,
map it to the right specialist type, then search the graph for the
shortest/cheapest path from Reception to that specialist.

Implements:
  - BFS (uninformed): shortest path by NUMBER OF HOPS, ignores edge weights
  - Uniform Cost Search / Dijkstra-style (uninformed, cost-aware): shortest
    path by TOTAL WEIGHT (e.g. distance/time between departments)

Both are shown so the report can compare "hops" vs "true shortest by cost" -
a classic uninformed-search demonstration (informed/A* would additionally
need a domain heuristic, which is discussed as a future enhancement in the
report since this small graph makes UCS already optimal and simple to reason
about).
"""

import json
import os
import heapq
from collections import deque

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "data", "specialist_graph.json")


class ClinicGraph:
    def __init__(self, graph_path=GRAPH_PATH):
        with open(graph_path, "r") as f:
            data = json.load(f)
        self.graph = data["graph"]
        self.condition_to_specialist = data["condition_to_specialist"]
        self.start_node = data["start_node"]

    def specialist_for(self, condition):
        return self.condition_to_specialist.get(condition, "General Physician")

    def bfs_path(self, start, goal):
        """Uninformed search: fewest hops, ignores weights."""
        if start == goal:
            return [start], 0
        visited = {start}
        queue = deque([(start, [start])])
        while queue:
            node, path = queue.popleft()
            for neighbor in self.graph.get(node, {}):
                if neighbor == goal:
                    return path + [neighbor], len(path)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None, None

    def uniform_cost_path(self, start, goal):
        """Uninformed but cost-aware search: true shortest path by total edge weight."""
        frontier = [(0, start, [start])]
        best_cost = {start: 0}
        while frontier:
            cost, node, path = heapq.heappop(frontier)
            if node == goal:
                return path, cost
            for neighbor, weight in self.graph.get(node, {}).items():
                new_cost = cost + weight
                if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                    best_cost[neighbor] = new_cost
                    heapq.heappush(frontier, (new_cost, neighbor, path + [neighbor]))
        return None, None

    def find_specialist_route(self, condition):
        target = self.specialist_for(condition)
        bfs_path, hops = self.bfs_path(self.start_node, target)
        ucs_path, cost = self.uniform_cost_path(self.start_node, target)
        return {
            "condition": condition,
            "specialist": target,
            "bfs_path": bfs_path,
            "bfs_hops": hops,
            "shortest_path": ucs_path,
            "shortest_cost": cost,
        }


if __name__ == "__main__":
    g = ClinicGraph()
    result = g.find_specialist_route("Respiratory Infection")
    print(result)
