# Imports --------------------------------------------------------------

import component1
import time
import random
import csv
import math
from collections import deque
import heapq
import os

import networkx as nx
import numpy as np
from simpleai.search import SearchProblem, breadth_first, depth_first, uniform_cost, astar, iterative_limited_depth_first

# Functions ------------------------------------------------------------

class GraphSearchProblem(SearchProblem):
    """Search problem adapter for SimpleAI library"""
    def __init__(self, G, start, goal, coords=None):
        self.graph = G
        self.goal_node = goal
        self.coords = coords  # For heuristic in A*
        super().__init__(initial_state=start)
    
    def actions(self, state):
        """Return list of neighbors"""
        return list(self.graph.neighbors(state))
    
    def result(self, state, action):
        """Action is the neighbor node to move to"""
        return action
    
    def is_goal(self, state):
        """Check if reached goal"""
        return state == self.goal_node
    
    def cost(self, state, action, state2):
        """Cost to move from state to state2"""
        return self.graph[state][action].get('length', 1.0)
    
    def heuristic(self, state):
        """Euclidean distance heuristic for A*"""
        if self.coords:
            return euclidean(self.coords[state], self.coords[self.goal_node])
        return 0

def euclidean(a, b):
    ax, ay = a
    bx, by = b
    return math.hypot(ax - bx, ay - by)

def build_simple_undirected_weighted(G):
    """
    Convert a NetworkX MultiDiGraph (OSMnx) to a simple undirected NetworkX Graph
    where edge weight 'length' is the minimum among parallel edges.
    Also return node->(x,y) coords dict.
    """
    if G.is_directed():
        Gu = nx.Graph()
    else:
        Gu = nx.Graph()

    # nodes
    coords = {}
    for n, data in G.nodes(data=True):
        # nodes must have 'x' and 'y' in projected graph
        if 'x' not in data or 'y' not in data:
            raise ValueError("Graph nodes must have 'x' and 'y' (projected coordinates)")
        coords[n] = (data['x'], data['y'])
        Gu.add_node(n)

    # edges: for MultiDiGraph choose minimum length among parallel edges
    for u, v, data in G.edges(data=True):
        length = data.get('length', None)
        if length is None:
            # fallback: straight-line distance using coords
            length = euclidean(coords[u], coords[v])
        if Gu.has_edge(u, v):
            # keep minimal length
            if length < Gu[u][v]['length']:
                Gu[u][v]['length'] = length
        else:
            Gu.add_edge(u, v, length=length)

    return Gu, coords

def bfs_search(G, start, goal):
    """Breadth-First Search using SimpleAI"""
    problem = GraphSearchProblem(G, start, goal)
    result = breadth_first(problem, graph_search=True)
    if result:
        # Extract states from path, filtering out any None values
        return [state for state, action in result.path() if state is not None]
    return None

def dfs_search(G, start, goal, max_nodes=10_000_000):
    """Depth-First Search using SimpleAI"""
    problem = GraphSearchProblem(G, start, goal)
    result = depth_first(problem, graph_search=True)
    if result:
        # Extract states from path, filtering out any None values
        return [state for state, action in result.path() if state is not None]
    return None

def ucs_search(G, start, goal):
    """Uniform Cost Search using SimpleAI"""
    problem = GraphSearchProblem(G, start, goal)
    result = uniform_cost(problem, graph_search=True)
    if result:
        # Extract states from path, filtering out any None values
        path = [state for state, action in result.path() if state is not None]
        # Calculate total cost
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += G[path[i]][path[i+1]].get('length', 1.0)
        return path, total_cost
    return None, float('inf')

def astar_search(G, start, goal, coords):
    """A* Search using SimpleAI"""
    problem = GraphSearchProblem(G, start, goal, coords)
    result = astar(problem, graph_search=True)
    if result:
        # Extract states from path, filtering out any None values
        path = [state for state, action in result.path() if state is not None]
        # Calculate total cost
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += G[path[i]][path[i+1]].get('length', 1.0)
        return path, total_cost
    return None, float('inf')

def iddfs_search(G, start, goal, max_depth=1000, timeout_seconds=None):
    """Iterative Deepening DFS using SimpleAI with timeout"""
    start_time = time.perf_counter() if timeout_seconds else None
    problem = GraphSearchProblem(G, start, goal)
    
    # SimpleAI's iterative_limited_depth_first already does iterative deepening
    # Just call it once with the max depth
    try:
        # Check if we should timeout before starting
        if timeout_seconds and (time.perf_counter() - start_time) > timeout_seconds:
            return None
        
        result = iterative_limited_depth_first(problem, depth_limit=max_depth, graph_search=True)
        if result:
            # Extract states from path, filtering out any None values
            return [state for state, action in result.path() if state is not None]
    except Exception as e:
        # If no solution found within depth limit
        return None
    
    return None

def select_pairs_by_distance(coords_dict, n_pairs=5, rng_seed=42):
    """
    coords_dict: {node: (x,y)}
    Returns three lists of pairs: small (<=1000), medium (1000-5000), large (>5000)
    Strategy: calculate all pairwise distances and categorize them efficiently.
    """
    random.seed(rng_seed)
    nodes = list(coords_dict.keys())
    
    # Calculate distances for a sample of node pairs to categorize
    small_pairs = []
    medium_pairs = []
    large_pairs = []
    
    # Use a smarter approach: sample random pairs and categorize them
    max_attempts = min(10000, len(nodes) * len(nodes))  # Limit attempts
    attempts = 0
    checked_pairs = set()
    
    while (len(small_pairs) < n_pairs or len(medium_pairs) < n_pairs or len(large_pairs) < n_pairs) and attempts < max_attempts:
        attempts += 1
        a = random.choice(nodes)
        b = random.choice(nodes)
        
        if a == b:
            continue
            
        # Create canonical pair representation
        pair = tuple(sorted([a, b]))
        if pair in checked_pairs:
            continue
        checked_pairs.add(pair)
        
        d = euclidean(coords_dict[a], coords_dict[b])
        
        # Categorize and add if needed
        if d <= 1000.0 and len(small_pairs) < n_pairs:
            small_pairs.append(pair)
        elif 1000.0 < d <= 5000.0 and len(medium_pairs) < n_pairs:
            medium_pairs.append(pair)
        elif d > 5000.0 and len(large_pairs) < n_pairs:
            large_pairs.append(pair)
    
    # If we couldn't find enough pairs in any category, warn the user
    if len(small_pairs) < n_pairs:
        print(f"Warning: Only found {len(small_pairs)} small pairs (requested {n_pairs})")
    if len(medium_pairs) < n_pairs:
        print(f"Warning: Only found {len(medium_pairs)} medium pairs (requested {n_pairs})")
    if len(large_pairs) < n_pairs:
        print(f"Warning: Only found {len(large_pairs)} large pairs (requested {n_pairs})")
    
    return small_pairs, medium_pairs, large_pairs

def run_experiments(G_proj, results_csv="route_eval_results.csv", n_pairs=5, rng_seed=42, timeout_seconds=10):
    """
    G_proj: projected OSMnx graph (NetworkX MultiDiGraph or Graph) already projected to metric coords.
    timeout_seconds: skip algorithm if it takes longer than this on any pair
    """
    # convert to simple undirected weighted graph
    print("Building simple undirected graph...")
    G_simple, coords = build_simple_undirected_weighted(G_proj)
    print(f"Nodes: {G_simple.number_of_nodes()}, Edges: {G_simple.number_of_edges()}")

    # choose node pairs
    print("Selecting pairs by euclidean distance...")
    small_pairs, medium_pairs, large_pairs = select_pairs_by_distance(coords, n_pairs=n_pairs, rng_seed=rng_seed)
    print(f"Found {len(small_pairs)} small, {len(medium_pairs)} medium, {len(large_pairs)} large pairs")
    
    # Filter out empty categories
    all_bins = []
    if small_pairs:
        all_bins.append(("small", small_pairs))
    if medium_pairs:
        all_bins.append(("medium", medium_pairs))
    if large_pairs:
        all_bins.append(("large", large_pairs))
    
    if not all_bins:
        print("Error: No valid node pairs found!")
        return None

    # prepare csv
    fieldnames = ["bin", "pair_index", "node_a", "node_b",
                  "euclidean_m",
                  "bfs_time_s", "bfs_len_nodes",
                  "dfs_time_s", "dfs_len_nodes",
                  "ucs_time_s", "ucs_cost_m", "ucs_len_nodes",
                  "iddfs_time_s", "iddfs_len_nodes",
                  "astar_time_s", "astar_cost_m", "astar_len_nodes"]
    if os.path.exists(results_csv):
        os.remove(results_csv)
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Track which algorithms to skip due to timeouts
        skip_algorithms = set()

        for bin_name, pairs in all_bins:
            for idx, (a, b) in enumerate(pairs):
                print(f"\n[{bin_name.upper()}] Processing pair {idx+1}/{len(pairs)}: nodes {a} -> {b}")
                row = {"bin": bin_name, "pair_index": idx, "node_a": a, "node_b": b}
                dist = euclidean(coords[a], coords[b])
                row["euclidean_m"] = dist
                print(f"  Euclidean distance: {dist:.1f} m")

                # BFS
                if "bfs" not in skip_algorithms:
                    print("  Running BFS...", end=" ", flush=True)
                    t0 = time.perf_counter()
                    path_bfs = bfs_search(G_simple, a, b)
                    t1 = time.perf_counter()
                    row["bfs_time_s"] = t1 - t0
                    row["bfs_len_nodes"] = len(path_bfs) if path_bfs else None
                    print(f"{row['bfs_time_s']:.6f}s")
                    if row["bfs_time_s"] > timeout_seconds:
                        print(f"  ⚠ BFS exceeded {timeout_seconds}s timeout - will skip for remaining pairs")
                        skip_algorithms.add("bfs")
                else:
                    print("  Skipping BFS (timeout)")
                    row["bfs_time_s"] = None
                    row["bfs_len_nodes"] = None

                # DFS
                if "dfs" not in skip_algorithms:
                    print("  Running DFS...", end=" ", flush=True)
                    t0 = time.perf_counter()
                    path_dfs = dfs_search(G_simple, a, b)
                    t1 = time.perf_counter()
                    row["dfs_time_s"] = t1 - t0
                    row["dfs_len_nodes"] = len(path_dfs) if path_dfs else None
                    print(f"{row['dfs_time_s']:.6f}s")
                    if row["dfs_time_s"] > timeout_seconds:
                        print(f"  ⚠ DFS exceeded {timeout_seconds}s timeout - will skip for remaining pairs")
                        skip_algorithms.add("dfs")
                else:
                    print("  Skipping DFS (timeout)")
                    row["dfs_time_s"] = None
                    row["dfs_len_nodes"] = None

                # UCS
                if "ucs" not in skip_algorithms:
                    print("  Running UCS...", end=" ", flush=True)
                    t0 = time.perf_counter()
                    path_ucs, cost_ucs = ucs_search(G_simple, a, b)
                    t1 = time.perf_counter()
                    row["ucs_time_s"] = t1 - t0
                    row["ucs_cost_m"] = cost_ucs if path_ucs else None
                    row["ucs_len_nodes"] = len(path_ucs) if path_ucs else None
                    print(f"{row['ucs_time_s']:.6f}s")
                    if row["ucs_time_s"] > timeout_seconds:
                        print(f"  ⚠ UCS exceeded {timeout_seconds}s timeout - will skip for remaining pairs")
                        skip_algorithms.add("ucs")
                else:
                    print("  Skipping UCS (timeout)")
                    row["ucs_time_s"] = None
                    row["ucs_cost_m"] = None
                    row["ucs_len_nodes"] = None

                # IDDFS - Test with built-in timeout
                if "iddfs" not in skip_algorithms:
                    # IDDFS is impractical for route planning - test but enforce timeout
                    if dist > 1000:  # Increased from 500 to 1000 to show it being used
                        print("  Skipping IDDFS (distance > 1000m)")
                        row["iddfs_time_s"] = None
                        row["iddfs_len_nodes"] = None
                    else:
                        print("  Running IDDFS...", end=" ", flush=True)
                        t0 = time.perf_counter()
                        # Pass timeout to the algorithm itself for early termination
                        path_iddfs = iddfs_search(G_simple, a, b, max_depth=500, timeout_seconds=timeout_seconds)
                        t1 = time.perf_counter()
                        row["iddfs_time_s"] = t1 - t0
                        row["iddfs_len_nodes"] = len(path_iddfs) if path_iddfs else None
                        
                        if row["iddfs_time_s"] >= timeout_seconds:
                            print(f"{row['iddfs_time_s']:.6f}s (TIMEOUT)")
                            print(f"  ⚠ IDDFS exceeded {timeout_seconds}s timeout - will skip for remaining pairs")
                            skip_algorithms.add("iddfs")
                        elif path_iddfs is None:
                            print(f"{row['iddfs_time_s']:.6f}s (no path found)")
                        else:
                            print(f"{row['iddfs_time_s']:.6f}s")
                else:
                    print("  Skipping IDDFS (timeout)")
                    row["iddfs_time_s"] = None
                    row["iddfs_len_nodes"] = None

                # A*
                if "astar" not in skip_algorithms:
                    print("  Running A*...", end=" ", flush=True)
                    t0 = time.perf_counter()
                    path_astar, cost_astar = astar_search(G_simple, a, b, coords)
                    t1 = time.perf_counter()
                    row["astar_time_s"] = t1 - t0
                    row["astar_cost_m"] = cost_astar if path_astar else None
                    row["astar_len_nodes"] = len(path_astar) if path_astar else None
                    print(f"{row['astar_time_s']:.6f}s")
                    if row["astar_time_s"] > timeout_seconds:
                        print(f"  ⚠ A* exceeded {timeout_seconds}s timeout - will skip for remaining pairs")
                        skip_algorithms.add("astar")
                else:
                    print("  Skipping A* (timeout)")
                    row["astar_time_s"] = None
                    row["astar_cost_m"] = None
                    row["astar_len_nodes"] = None

                writer.writerow(row)

    print("Done. Results saved to", results_csv)
    return results_csv

# Main -----------------------------------------------------------------

if __name__ == "__main__":
    # Load the graph using component1's function
    print("Loading campus graph...")
    G, G_proj = component1.load_campus_graph(
        campus_name='Tec de Monterrey campus Guadalajara, Zapopan, Jalisco, 45201, México',
        buffer_meters=6000,
        network_type="walk"
    )
    
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Projected graph: {G_proj.number_of_nodes()} nodes")
    
    # Run route planning experiments
    print("\nRunning route planning experiments...")
    results_file = run_experiments(
        G_proj, 
        results_csv="route_eval_results.csv", 
        n_pairs=5, 
        rng_seed=42
    )
    
    print(f"\nExperiments complete! Results saved to: {results_file}")
