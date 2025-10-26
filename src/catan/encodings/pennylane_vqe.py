from __future__ import annotations

import time
import math
import numpy as np
import pennylane as qml
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# ============================================================
# Conversie QUBO → Hamiltonian Ising
# ============================================================

def qubo_to_hamiltonian(Q: np.ndarray) -> Tuple[qml.Hamiltonian, float]:
    """
    Convertește QUBO (min x^T Q x) într-un Hamiltonian Ising pentru PennyLane.
    x_i = (1 - z_i)/2 → operatorii Z_i și Z_i Z_j.
    Returnează (Hamiltonian, offset).
    """
    n = Q.shape[0]
    coeffs = []
    ops = []
    offset = 0.0

    for i in range(n):
        offset += Q[i, i] / 4
        coeffs.append(-Q[i, i] / 2)
        ops.append(qml.PauliZ(i))
        for j in range(i + 1, n):
            q = Q[i, j]
            if abs(q) < 1e-12:
                continue
            offset += q / 4
            coeffs.append(q / 4)
            ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
            coeffs.append(-q / 4)
            ops.append(qml.PauliZ(i))
            coeffs.append(-q / 4)
            ops.append(qml.PauliZ(j))
    H = qml.Hamiltonian(coeffs, ops)
    return H, offset


# ============================================================
# Rezultat structurat
# ============================================================

@dataclass
class VQEResult:
    params: np.ndarray
    energy: float
    runtime: float
    bitstring: str
    probabilities: Dict[str, float]


# ============================================================
# Runner principal
# ============================================================

def run_vqe_qubo(
    Q: np.ndarray,
    layers: int = 2,
    optimizer: str = "Adam",
    steps: int = 200,
    lr: float = 0.1,
    backend: str = "default.qubit",
    shots: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Rulează VQE pentru un QUBO dat folosind PennyLane.
    Returnează dict cu energia minimă și parametrii optimi.

    Parametri:
      - Q: matrice QUBO (numpy, n x n)
      - layers: numărul de straturi de ansatz
      - optimizer: 'Adam', 'GD', 'NelderMead'
      - steps: pași de optimizare
      - lr: rata de învățare
      - backend: simulator PennyLane
      - shots: None = analitic, altfel sampling
      - seed: reproducibilitate
    """
    np.random.seed(seed)
    start = time.time()
    n = Q.shape[0]

    H, offset = qubo_to_hamiltonian(Q)

    if verbose:
        print(f"[VQE] n={n}, layers={layers}, optimizer={optimizer}, backend={backend}")

    # Device
    dev = qml.device(backend, wires=n, shots=shots)

    # Ansatz parametric
    def ansatz(params):
        for l in range(layers):
            for i in range(n):
                qml.RY(params[l, i], wires=i)
            for i in range(n - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n - 1, 0])

    @qml.qnode(dev)
    def circuit(params):
        ansatz(params)
        return qml.expval(H)

    # Inițializare parametri
    params = 2 * np.pi * np.random.rand(layers, n)

    # Alegere optimizer
    if optimizer.lower() == "adam":
        opt = qml.GradientDescentOptimizer(stepsize=lr)
    elif optimizer.lower() in ("gd", "grad"):
        opt = qml.GradientDescentOptimizer(stepsize=lr)
    elif optimizer.lower() in ("nelder", "nm"):
        from scipy.optimize import minimize

        def objective(x):
            return circuit(x.reshape(layers, n))

        res = minimize(objective, params.flatten(), method="Nelder-Mead", options={"maxiter": steps})
        params = res.x.reshape(layers, n)
        minE = res.fun
        return {
            "energy": minE + offset,
            "params": params,
            "runtime": time.time() - start,
            "n_qubits": n,
        }
    else:
        opt = qml.AdamOptimizer(stepsize=lr)

    # Optimizare
    energy_history = []
    for it in range(steps):
        params, E = opt.step_and_cost(circuit, params)
        energy_history.append(E)
        if verbose and it % max(1, steps // 10) == 0:
            print(f"  step {it:4d}  ⟶  E = {E + offset:.6f}")

    runtime = time.time() - start
    final_E = energy_history[-1] + offset

    # Măsurăm distribuția probabilităților
    @qml.qnode(dev)
    def measure(params):
        ansatz(params)
        return qml.probs(wires=range(n))

    probs = measure(params)
    keys = [format(i, f"0{n}b")[::-1] for i in range(2**n)]
    prob_dict = {k: float(p) for k, p in zip(keys, probs)}
    best_bs = max(prob_dict, key=prob_dict.get)
    x = np.array([int(b) for b in best_bs[::-1]])
    best_energy = float(x @ Q @ x)

    if verbose:
        print(f"[VQE] Done in {runtime:.2f}s | Best bitstring: {best_bs} | Energy: {best_energy:.6f}")

    return {
        "params": params,
        "energy": final_E,
        "runtime": runtime,
        "best_bitstring": best_bs,
        "best_energy": best_energy,
        "probabilities": prob_dict,
        "n_qubits": n,
        "offset": offset,
        "energy_history": energy_history,
    }


# ============================================================
# Plot helpers
# ============================================================

def plot_vqe_convergence(energy_history: List[float]):
    plt.figure(figsize=(6, 3))
    plt.plot(range(len(energy_history)), energy_history, color="royalblue")
    plt.title("VQE Energy Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Energy (no offset)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_vqe_probabilities(prob_dict: Dict[str, float], Q: np.ndarray):
    energies = []
    probs = []
    bitstrings = []

    for bs, p in prob_dict.items():
        x = np.array([int(b) for b in bs[::-1]])
        e = float(x @ Q @ x)
        energies.append(e)
        probs.append(p)
        bitstrings.append(bs)

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(bitstrings)), probs, color="mediumseagreen", alpha=0.7)
    plt.title("Bitstring Probabilities (VQE final state)")
    plt.xlabel("Bitstring index")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# Test local
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)
    Q = np.array([
        [-1.0, 0.5, 0.0],
        [0.5, -1.0, 0.5],
        [0.0, 0.5, -1.0],
    ])
    res = run_vqe_qubo(Q, layers=2, steps=100, lr=0.2, optimizer="adam", verbose=True)
    plot_vqe_convergence(res["energy_history"])
    plot_vqe_probabilities(res["probabilities"], Q)
