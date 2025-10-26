"""
Problem 2: Quantum Longest Road
================================
Find the longest connected road path using QAOA optimization.

Strategy:
- Start with initial roads for ally and adversary
- Select up to K new roads to build (resource constraint)
- Maximize the longest connected path length for ally
- New roads must connect to existing ally roads
- Adversary roads block certain edges
"""

from __future__ import annotations
import numpy as np
from typing import List, Set, Tuple, Dict, Optional
from collections import deque
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
try:
    from qiskit_aer.primitives import Sampler
except ImportError:
    try:
        from qiskit.primitives import StatevectorSampler as Sampler
    except ImportError:
        from qiskit.primitives import BackendSampler
        Sampler = lambda: BackendSampler(backend=AerSimulator())

# Local imports
from catan.board import Board, generate_classic_board
from catan.viz import plot_board, highlight_edges
from catan.utils import Timer, print_section, cprint, Colors, show_or_save, set_seed


# ===============================
# Graph utilities
# ===============================

def find_longest_path_dfs(board: Board, edge_set: Set[int]) -> int:
    """
    Find longest path in undirected graph using DFS.
    Returns the maximum path length (number of edges).
    """
    if not edge_set:
        return 0
    
    # Build adjacency from edges
    adj: Dict[int, Set[int]] = {}
    for eid in edge_set:
        e = board.edges[eid]
        if e.u not in adj:
            adj[e.u] = set()
        if e.v not in adj:
            adj[e.v] = set()
        adj[e.u].add(e.v)
        adj[e.v].add(e.u)
    
    max_length = 0
    
    def dfs(node: int, visited: Set[int], length: int):
        nonlocal max_length
        max_length = max(max_length, length)
        
        for neighbor in adj.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                dfs(neighbor, visited, length + 1)
                visited.remove(neighbor)
    
    # Try starting from each node
    for start_node in adj.keys():
        visited = {start_node}
        dfs(start_node, visited, 0)
    
    return max_length


def get_connected_component(board: Board, edge_set: Set[int], seed_edges: Set[int]) -> Set[int]:
    """
    Get all edges connected to seed_edges via BFS.
    Returns set of edge IDs that form connected component.
    """
    if not seed_edges:
        return edge_set.copy()
    
    # Build node adjacency via edges
    edge_at_nodes: Dict[Tuple[int,int], int] = {}
    for eid in edge_set:
        e = board.edges[eid]
        edge_at_nodes[(min(e.u, e.v), max(e.u, e.v))] = eid
    
    # Start from nodes in seed edges
    visited_edges = seed_edges.copy()
    queue = deque()
    visited_nodes = set()
    
    for eid in seed_edges:
        e = board.edges[eid]
        queue.append(e.u)
        queue.append(e.v)
        visited_nodes.add(e.u)
        visited_nodes.add(e.v)
    
    # BFS to find connected edges
    while queue:
        node = queue.popleft()
        
        # Check all neighbors
        for neighbor in board.node_neighbors.get(node, []):
            edge_key = (min(node, neighbor), max(node, neighbor))
            
            if edge_key in edge_at_nodes:
                eid = edge_at_nodes[edge_key]
                
                if eid in edge_set and eid not in visited_edges:
                    visited_edges.add(eid)
                    
                    if neighbor not in visited_nodes:
                        visited_nodes.add(neighbor)
                        queue.append(neighbor)
    
    return visited_edges


# ===============================
# QUBO formulation
# ===============================

def build_longest_road_qubo(
    board: Board,
    candidate_edges: List[int],
    initial_ally_edges: Set[int],
    blocked_edges: Set[int],
    max_new_roads: int,
    penalty_disconnected: float = 10.0,
    penalty_budget: float = 8.0,
    reward_length: float = 3.0,
) -> np.ndarray:
    """
    Build QUBO matrix for longest road problem.
    
    Decision variables: x_i = 1 if candidate edge i is built
    
    Objective:
    - Maximize: length of longest path in (initial_ally ∪ selected_candidates)
    - Penalize: edges not connected to initial_ally
    - Constrain: at most max_new_roads edges selected
    
    Returns: Q matrix where x^T Q x is the objective (to minimize)
    """
    n = len(candidate_edges)
    Q = np.zeros((n, n))
    
    # Build node connectivity from initial roads
    ally_nodes = set()
    for eid in initial_ally_edges:
        e = board.edges[eid]
        ally_nodes.add(e.u)
        ally_nodes.add(e.v)
    
    # For each candidate edge, check connectivity and contribution
    for i, eid_i in enumerate(candidate_edges):
        e_i = board.edges[eid_i]
        
        # Reward if edge connects to ally network
        connected_to_ally = (e_i.u in ally_nodes) or (e_i.v in ally_nodes)
        
        if connected_to_ally:
            # Reward selecting this edge (helps extend road)
            Q[i, i] -= reward_length
        else:
            # Penalize disconnected edges
            Q[i, i] += penalty_disconnected
        
        # Interaction terms: edges sharing a node reinforce each other
        for j in range(i + 1, n):
            eid_j = candidate_edges[j]
            e_j = board.edges[eid_j]
            
            # If edges share a node, reward selecting both (builds longer paths)
            shares_node = len({e_i.u, e_i.v} & {e_j.u, e_j.v}) > 0
            
            if shares_node:
                Q[i, j] -= reward_length * 0.5
                Q[j, i] = Q[i, j]
    
    # Budget constraint: penalize selecting too many edges
    # (sum x_i - K)^2 = sum x_i^2 - 2K sum x_i + K^2
    for i in range(n):
        Q[i, i] += penalty_budget
        for j in range(i + 1, n):
            Q[i, j] += 2 * penalty_budget
            Q[j, i] = Q[i, j]
    
    for i in range(n):
        Q[i, i] -= 2 * penalty_budget * max_new_roads
    
    return Q


# ===============================
# QAOA implementation
# ===============================

def qaoa_circuit(Q: np.ndarray, gamma: float, beta: float) -> QuantumCircuit:
    """
    Build QAOA circuit for QUBO problem.
    
    Phase separator: exp(-i γ H_C) where H_C encodes QUBO
    Mixer: exp(-i β H_M) where H_M = sum X_i
    """
    n = Q.shape[0]
    qc = QuantumCircuit(n)
    
    # Initial superposition
    qc.h(range(n))
    
    # Phase separator (problem Hamiltonian)
    for i in range(n):
        if Q[i, i] != 0:
            qc.rz(2 * gamma * Q[i, i], i)
    
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                qc.cx(i, j)
                qc.rz(2 * gamma * Q[i, j], j)
                qc.cx(i, j)
    
    # Mixer (X rotation on all qubits)
    qc.rx(2 * beta, range(n))
    
    # Measurement
    qc.measure_all()
    
    return qc


def evaluate_solution(
    board: Board,
    selected_edges: List[int],
    initial_ally_edges: Set[int],
) -> int:
    """Evaluate solution: longest path length in combined road network."""
    all_edges = initial_ally_edges | set(selected_edges)
    return find_longest_path_dfs(board, all_edges)


def optimize_qaoa(
    Q: np.ndarray,
    board: Board,
    candidate_edges: List[int],
    initial_ally_edges: Set[int],
    p: int = 1,
    shots: int = 1024,
    max_iter: int = 50,
) -> Tuple[List[int], int, Dict]:
    """
    Optimize QAOA parameters and return best solution.
    
    Returns: (best_edges, best_length, info_dict)
    """
    n = Q.shape[0]
    
    # Simple grid search for parameters (for demo purposes)
    best_solution = []
    best_length = 0
    best_params = None
    
    gamma_range = np.linspace(0, np.pi, 5)
    beta_range = np.linspace(0, np.pi, 5)
    
    sampler = Sampler()
    
    for gamma in gamma_range:
        for beta in beta_range:
            # Build and run circuit
            qc = qaoa_circuit(Q, gamma, beta)
            
            # Sample
            job = sampler.run(qc, shots=shots)
            result = job.result()
            counts = result.quasi_dists[0]
            
            # Evaluate top solutions
            for bitstring_int, prob in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                bitstring = format(bitstring_int, f'0{n}b')[::-1]  # Reverse for qubit ordering
                selected = [candidate_edges[i] for i in range(n) if bitstring[i] == '1']
                
                # Check connectivity
                all_edges = initial_ally_edges | set(selected)
                connected = get_connected_component(board, all_edges, initial_ally_edges)
                
                # Only consider connected solutions
                if len(connected) == len(all_edges):
                    length = find_longest_path_dfs(board, all_edges)
                    
                    if length > best_length:
                        best_length = length
                        best_solution = selected
                        best_params = (gamma, beta)
    
    info = {
        'best_params': best_params,
        'n_qubits': n,
        'shots': shots,
    }
    
    return best_solution, best_length, info


# ===============================
# Classical baseline
# ===============================

def greedy_road_extension(
    board: Board,
    candidate_edges: List[int],
    initial_ally_edges: Set[int],
    max_new_roads: int,
) -> Tuple[List[int], int]:
    """
    Greedy baseline: iteratively add edge that maximizes path length.
    """
    current_edges = initial_ally_edges.copy()
    selected = []
    
    for _ in range(max_new_roads):
        best_edge = None
        best_length = find_longest_path_dfs(board, current_edges)
        
        for eid in candidate_edges:
            if eid in current_edges or eid in selected:
                continue
            
            # Check if edge connects to current network
            e = board.edges[eid]
            ally_nodes = set()
            for e2_id in current_edges:
                e2 = board.edges[e2_id]
                ally_nodes.add(e2.u)
                ally_nodes.add(e2.v)
            
            if e.u not in ally_nodes and e.v not in ally_nodes:
                continue
            
            # Try adding this edge
            test_edges = current_edges | {eid}
            test_length = find_longest_path_dfs(board, test_edges)
            
            if test_length > best_length:
                best_length = test_length
                best_edge = eid
        
        if best_edge is None:
            break
        
        selected.append(best_edge)
        current_edges.add(best_edge)
    
    final_length = find_longest_path_dfs(board, current_edges)
    return selected, final_length


# ===============================
# Scenario generation
# ===============================

def generate_scenario(board: Board, seed: Optional[int] = None) -> Tuple[Set[int], Set[int]]:
    """
    Generate initial ally and adversary roads.
    
    Returns: (initial_ally_edges, blocked_adversary_edges)
    """
    rng = np.random.RandomState(seed)
    
    # Pick random starting edges for ally (3-5 edges)
    n_initial_ally = rng.randint(3, 6)
    ally_edges = set()
    
    # Start from a random edge
    start_edge = rng.choice(board.edges)
    ally_edges.add(start_edge.id)
    
    # Grow connected component
    frontier = {start_edge.u, start_edge.v}
    
    while len(ally_edges) < n_initial_ally and frontier:
        node = frontier.pop()
        
        # Find edges from this node
        candidates = []
        for eid, e in enumerate(board.edges):
            if eid not in ally_edges:
                if e.u == node or e.v == node:
                    candidates.append(eid)
        
        if candidates:
            new_edge_id = rng.choice(candidates)
            ally_edges.add(new_edge_id)
            
            e = board.edges[new_edge_id]
            frontier.add(e.u)
            frontier.add(e.v)
    
    # Generate adversary roads (blocking some edges)
    n_blocked = rng.randint(4, 8)
    available = [e.id for e in board.edges if e.id not in ally_edges]
    blocked = set(rng.choice(available, size=min(n_blocked, len(available)), replace=False))
    
    return ally_edges, blocked


# ===============================
# Main solver
# ===============================

def solve_longest_road(
    n_qubits: int = 12,
    max_new_roads: int = 6,
    seed: Optional[int] = None,
    use_qaoa: bool = True,
    visualize: bool = True,
) -> Dict:
    """
    Solve the longest road problem.
    
    Args:
        n_qubits: Number of candidate edges to consider
        max_new_roads: Resource constraint (max roads to build)
        seed: Random seed
        use_qaoa: Whether to use QAOA (else only classical)
        visualize: Whether to generate plots
    
    Returns: Dictionary with results
    """
    if seed is not None:
        set_seed(seed)
    
    print_section("Problem 2: Quantum Longest Road")
    cprint(f"Configuration: {n_qubits} qubits, max {max_new_roads} new roads", Colors.CYAN)
    
    # Generate board and scenario
    board = generate_classic_board(seed=seed)
    initial_ally, blocked = generate_scenario(board, seed=seed)
    
    print(f"Initial ally roads: {len(initial_ally)}")
    print(f"Blocked (adversary) roads: {len(blocked)}")
    
    # Select candidate edges (exclude initial and blocked)
    available_edges = [
        e.id for e in board.edges 
        if e.id not in initial_ally and e.id not in blocked
    ]
    
    # Score by connectivity to ally nodes
    ally_nodes = set()
    for eid in initial_ally:
        e = board.edges[eid]
        ally_nodes.add(e.u)
        ally_nodes.add(e.v)
    
    edge_scores = []
    for eid in available_edges:
        e = board.edges[eid]
        # Prefer edges near ally network
        score = 0
        if e.u in ally_nodes:
            score += 2
        if e.v in ally_nodes:
            score += 2
        # Prefer edges with high resource value
        node_values = []
        for nid in [e.u, e.v]:
            val = sum(
                board.tiles[tid].number or 0 
                for tid in board.node_tiles.get(nid, [])
            )
            node_values.append(val)
        score += sum(node_values) / 20.0
        edge_scores.append((score, eid))
    
    edge_scores.sort(reverse=True)
    candidate_edges = [eid for _, eid in edge_scores[:n_qubits]]
    
    print(f"Selected {len(candidate_edges)} candidate edges")
    
    # Initial length
    initial_length = find_longest_path_dfs(board, initial_ally)
    print(f"Initial longest path: {initial_length} edges")
    
    results = {
        'board': board,
        'initial_ally': initial_ally,
        'blocked': blocked,
        'candidate_edges': candidate_edges,
        'initial_length': initial_length,
    }
    
    # Classical baseline
    print_section("Classical Baseline (Greedy)")
    with Timer("Greedy"):
        greedy_selected, greedy_length = greedy_road_extension(
            board, candidate_edges, initial_ally, max_new_roads
        )
    
    print(f"Greedy solution: selected {len(greedy_selected)} roads → length {greedy_length}")
    results['greedy_selected'] = greedy_selected
    results['greedy_length'] = greedy_length
    
    # QAOA solution
    if use_qaoa:
        print_section("QAOA Optimization")
        
        with Timer("Build QUBO"):
            Q = build_longest_road_qubo(
                board, candidate_edges, initial_ally, blocked, max_new_roads
            )
        
        print(f"QUBO matrix: {Q.shape}")
        
        with Timer("QAOA"):
            qaoa_selected, qaoa_length, info = optimize_qaoa(
                Q, board, candidate_edges, initial_ally, p=1, shots=1024
            )
        
        print(f"QAOA solution: selected {len(qaoa_selected)} roads → length {qaoa_length}")
        print(f"Best params: {info['best_params']}")
        
        results['qaoa_selected'] = qaoa_selected
        results['qaoa_length'] = qaoa_length
        results['qaoa_info'] = info
    
    # Visualization
    if visualize:
        print_section("Visualization")
        
        # Overview
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Initial state
        plot_board(board, ax=axes[0])
        highlight_edges(board, initial_ally, ax=axes[0], color='blue', width=3.0, label=f'Ally initial ({len(initial_ally)})')
        highlight_edges(board, blocked, ax=axes[0], color='red', width=3.0, label=f'Adversary ({len(blocked)})')
        axes[0].set_title(f'Initial State\nLongest path: {initial_length}')
        axes[0].legend()
        
        # Greedy solution
        plot_board(board, ax=axes[1])
        highlight_edges(board, initial_ally, ax=axes[1], color='blue', width=2.5)
        highlight_edges(board, greedy_selected, ax=axes[1], color='green', width=3.5, label=f'Greedy new ({len(greedy_selected)})')
        highlight_edges(board, blocked, ax=axes[1], color='red', width=2.0, label=f'Adversary')
        axes[1].set_title(f'Greedy Solution\nLongest path: {greedy_length} (+{greedy_length - initial_length})')
        axes[1].legend()
        
        # QAOA solution
        if use_qaoa:
            plot_board(board, ax=axes[2])
            highlight_edges(board, initial_ally, ax=axes[2], color='blue', width=2.5)
            highlight_edges(board, qaoa_selected, ax=axes[2], color='purple', width=3.5, label=f'QAOA new ({len(qaoa_selected)})')
            highlight_edges(board, blocked, ax=axes[2], color='red', width=2.0, label=f'Adversary')
            axes[2].set_title(f'QAOA Solution\nLongest path: {qaoa_length} (+{qaoa_length - initial_length})')
            axes[2].legend()
        else:
            axes[2].axis('off')
            axes[2].text(0.5, 0.5, 'QAOA disabled', ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        show_or_save(fig, "p2_longest_road_comparison.png")
        
        results['figure'] = fig
    
    # Summary
    print_section("Summary")
    print(f"Initial:  {initial_length} edges")
    print(f"Greedy:   {greedy_length} edges (+{greedy_length - initial_length})")
    if use_qaoa:
        print(f"QAOA:     {qaoa_length} edges (+{qaoa_length - initial_length})")
        improvement = qaoa_length - greedy_length
        if improvement > 0:
            cprint(f"✓ QAOA outperforms greedy by {improvement} edges!", Colors.GREEN)
        elif improvement == 0:
            cprint(f"= QAOA matches greedy performance", Colors.YELLOW)
        else:
            cprint(f"✗ QAOA underperforms greedy by {-improvement} edges", Colors.RED)
    
    return results


# ===============================
# Entry point
# ===============================

if __name__ == "__main__":
    results = solve_longest_road(
        n_qubits=12,
        max_new_roads=7,
        seed=20,
        use_qaoa=True,
        visualize=True,
    )
    
    print("\n" + "="*50)
    print("Problem 2 completed successfully!")
    print("="*50)