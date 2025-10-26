from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from catan.utils import set_seed, print_section, cprint, Colors, show_or_save
from catan.board import board_for_qubits
from catan.viz import plot_board, highlight_nodes
from catan.encodings.qubo import build_settlement_qubo, _weights_settlement
from catan.encodings.qiskit_qaoa import run_qaoa_qubo, plot_energy_landscape
from catan.encodings.pennylane_vqe import run_vqe_qubo, plot_vqe_convergence, plot_vqe_probabilities


def _decode_best(bitstring: str, n: int) -> str:
    """Keep exactly n least-significant bits (rightmost), pad if shorter."""
    if len(bitstring) >= n:
        return bitstring[-n:]
    return bitstring.zfill(n)


def _best_feasible_pair(board, selected_nodes, weights) -> list[int]:
    """Classical fallback: choose the best non-adjacent pair by weight."""
    # Build adjacency on candidate set
    n = len(selected_nodes)
    id_to_idx = {nid: i for i, nid in enumerate(selected_nodes)}
    neighbors = {nid: set() for nid in selected_nodes}
    for nid in selected_nodes:
        for nb in getattr(board, "node_neighbors", {}).get(nid, []):
            if nb in id_to_idx:
                neighbors[nid].add(nb)

    best_score, best_pair = -1.0, []
    for a_idx in range(n):
        a = selected_nodes[a_idx]
        for b_idx in range(a_idx + 1, n):
            b = selected_nodes[b_idx]
            if (b in neighbors[a]) or (a in neighbors[b]):
                continue  # adjacent → not allowed
            score = weights[a_idx] + weights[b_idx]
            if score > best_score:
                best_score, best_pair = score, [a, b]
    return best_pair


def main():
    parser = argparse.ArgumentParser(description="Quantum Catan — Problem 1: Settlements (exactly two)")
    parser.add_argument("--qubits", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="qaoa", choices=["qaoa", "vqe"])
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="COBYLA")
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--A", type=float, default=None)
    parser.add_argument("--mu-two", type=float, default=None)
    parser.add_argument("--no-reward", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    print_section(f"P1 – Settlements | N_QUBITS={args.qubits} | METHOD={args.method.upper()} | EXACT=2")

    # Candidates
    board, selected_nodes, _ = board_for_qubits(n_qubits=args.qubits, mode="settlement", seed=args.seed)
    n = len(selected_nodes)
    cprint(f"Selected {n} candidate nodes", Colors.CYAN)

    # Candidate figure
    fig, ax = plot_board(board)
    highlight_nodes(board, selected_nodes, ax=ax, color=(0.10, 0.55, 0.95), size=90, label=f"{n} nodes")
    ax.set_title(f"Candidate settlements ({n})")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper right")
    show_or_save(fig, "p1_candidates.png")

    # Build QUBO with EXACT TWO
    model = build_settlement_qubo(
        board,
        selected_nodes,
        A=args.A,
        maximize_weight=not args.no_reward,
        k_exact=2,
        mu_exact=args.mu_two,  # strong default if None inside builder
    )
    Q = model.Q
    cprint(f"QUBO matrix {Q.shape} | nnz={np.count_nonzero(Q)}", Colors.CYAN)

    # Solve
    if args.method == "qaoa":
        res = run_qaoa_qubo(Q, reps=args.reps, optimizer=args.optimizer, shots=args.shots, seed=args.seed)
    else:
        res = run_vqe_qubo(Q, layers=2, steps=args.steps, lr=args.lr, optimizer="adam")

    # Robust decoding to exactly n bits (LSB)
    raw_bits = _decode_best(res["best_bitstring"], n)
    energy   = res["best_energy"]
    cprint(f"[raw] best bitstring={raw_bits} | energy={energy:.6f}", Colors.GREEN)

    chosen_idx = [i for i, b in enumerate(raw_bits[::-1]) if b == "1"]
    chosen_nodes = [model.id_of[i] for i in chosen_idx]

    # If solver didn't return exactly 2 non-adjacent, project to the best feasible pair
    if len(chosen_nodes) != 2:
        w = _weights_settlement(board, selected_nodes)
        chosen_nodes = _best_feasible_pair(board, selected_nodes, w)
        cprint(f"[fallback] projected to best feasible pair: {chosen_nodes}", Colors.YELLOW)
    else:
        cprint(f"Chosen nodes (exactly 2): {chosen_nodes}", Colors.CYAN)

    # Draw solution
    fig, ax = plot_board(board)
    highlight_nodes(board, selected_nodes, ax=ax, color=(0.10, 0.55, 0.95), size=70, label="Candidates", lw=1.2)
    highlight_nodes(board, chosen_nodes,   ax=ax, color=(0.95, 0.15, 0.15), size=140, label="Chosen (2)", lw=2.4)
    ax.set_title(f"P1 – Optimal Settlements ({args.method.upper()})")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper right")
    show_or_save(fig, "p1_solution.png")

    # Diagnostics
    if args.method == "qaoa":
        plot_energy_landscape(res["bitstrings"], Q)
    else:
        plot_vqe_convergence(res["energy_history"])
        plot_vqe_probabilities(res["probabilities"], Q)

    # Save artefacts
    if args.save:
        out = Path("data"); out.mkdir(exist_ok=True)
        np.save(out / "p1_Q.npy", Q)
        with open(out / "p1_best.txt", "w") as f:
            f.write(
                f"method={args.method}\n"
                f"bitstring={raw_bits}\n"
                f"energy={energy}\n"
                f"chosen_nodes={chosen_nodes}\n"
                f"A={args.A}, mu_two={args.mu_two}\n"
            )
        cprint("[Saved] data/p1_Q.npy, data/p1_best.txt and figures", Colors.GREEN)


if __name__ == "__main__":
    main()
