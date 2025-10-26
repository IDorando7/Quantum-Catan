from __future__ import annotations
import numpy as np, time
from dataclasses import dataclass
from typing import Dict, Tuple
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from catan.utils import show_or_save


# ============================================================
# QUBO → Ising
# ============================================================
def _qubo_to_ising(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    n = Q.shape[0]
    h, J = np.zeros(n), np.zeros((n, n))
    offset = 0.0
    for i in range(n):
        offset += Q[i, i] / 4.0
        h[i] += -Q[i, i] / 2.0
        for j in range(i + 1, n):
            q = Q[i, j]
            if q != 0.0:
                J[i, j] += q / 4.0
                offset += q / 4.0
                h[i] += -q / 4.0
                h[j] += -q / 4.0
    return h, J, offset


# ============================================================
# QAOA Circuit and Evaluation
# ============================================================
def _qaoa_circuit(params: np.ndarray, h: np.ndarray, J: np.ndarray) -> QuantumCircuit:
    n = len(h)
    p = len(params) // 2
    gammas, betas = params[:p], params[p:]
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for layer in range(p):
        g, b = gammas[layer], betas[layer]
        for i in range(n):
            if abs(h[i]) > 1e-9:
                qc.rz(2 * g * h[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-9:
                    qc.cx(i, j); qc.rz(2 * g * J[i, j], j); qc.cx(i, j)
        for i in range(n):
            qc.rx(2 * b, i)
    qc.measure_all()
    return qc


def _expected_energy(counts: Dict[str, int], Q: np.ndarray, shots: int) -> float:
    E = 0.0
    for bitstring, c in counts.items():
        x = np.array([int(b) for b in bitstring[::-1]])
        E += (x @ Q @ x) * (c / shots)
    return E


@dataclass
class QAOAResult:
    bitstrings: Dict[str, float]
    best_bitstring: str
    best_energy: float
    energy_exp: float
    runtime: float
    n_qubits: int


# ============================================================
# Runner using AerSimulator
# ============================================================
def run_qaoa_qubo(Q: np.ndarray, reps=1, shots=512, optimizer="COBYLA", seed=42, verbose=True) -> Dict[str, any]:
    np.random.seed(seed)
    start = time.time()
    n = Q.shape[0]
    h, J, _ = _qubo_to_ising(Q)
    sim = AerSimulator(seed_simulator=seed)

    def objective(params):
        circ = _qaoa_circuit(params, h, J)
        job = sim.run(circ, shots=shots)
        return _expected_energy(job.result().get_counts(), Q, shots)

    x0 = 0.01 * np.random.randn(2 * reps)
    res = minimize(objective, x0, method="COBYLA", options={"maxiter": 80})
    circ = _qaoa_circuit(res.x, h, J)
    counts = sim.run(circ, shots=shots).result().get_counts()
    probs = {k: v / shots for k, v in counts.items()}
    best_bs = max(probs, key=probs.get)
    x = np.array([int(b) for b in best_bs[::-1]])
    best_energy = float(x @ Q @ x)
    exp_E = _expected_energy(counts, Q, shots)
    runtime = time.time() - start
    if verbose:
        print(f"[QAOA] Done in {runtime:.2f}s | Best={best_bs}, Energy={best_energy:.6f}")
    return {"bitstrings": probs, "best_bitstring": best_bs, "best_energy": best_energy,
            "exp_value": exp_E, "runtime": runtime, "n_qubits": n}


# ============================================================
# Plots with show_or_save
# ============================================================
def plot_energy_landscape(probabilities: Dict[str, float], Q: np.ndarray):
    energies, probs, bitstrings = [], [], []
    for bs, p in probabilities.items():
        x = np.array([int(b) for b in bs[::-1]])
        energies.append(float(x @ Q @ x)); probs.append(p); bitstrings.append(bs)

    fig = plt.figure(figsize=(8, 4))
    plt.bar(range(len(bitstrings)), energies, color="lightcoral", alpha=0.7)
    plt.title("Energy per Bitstring"); plt.xlabel("Bitstring index"); plt.ylabel("Energy")
    plt.grid(True, alpha=0.3)
    show_or_save(fig, "qaoa_energy.png")

    fig = plt.figure(figsize=(8, 4))
    plt.bar(range(len(bitstrings)), probs, color="skyblue", alpha=0.7)
    plt.title("Bitstring Probability Distribution"); plt.xlabel("Bitstring index"); plt.ylabel("Probability")
    plt.grid(True, alpha=0.3)
    show_or_save(fig, "qaoa_probabilities.png")
