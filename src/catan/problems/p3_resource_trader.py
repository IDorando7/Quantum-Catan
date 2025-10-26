from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from catan.utils import set_seed, print_section, cprint, Colors, show_or_save
from catan.encodings.qiskit_qaoa import run_qaoa_qubo, plot_energy_landscape
from catan.encodings.pennylane_vqe import run_vqe_qubo


# ==============================================================
# 1️⃣ Generare aleatoare de trade-uri realiste
# ==============================================================

RESOURCES = ["wood", "brick", "sheep", "wheat", "ore"]


def generate_random_trades(n_trades: int = 8, seed: int = 42) -> list[dict]:
    """
    Generează o listă de trade-uri aleatoare, realiste:
      - 'give' și 'get' diferite
      - rate ∈ {2, 3, 4}
      - value aleator în [1.0, 2.0] (valoare strategică)
    """
    rng = np.random.default_rng(seed)
    trades = []
    for _ in range(n_trades):
        give, get = rng.choice(RESOURCES, size=2, replace=False)
        rate = float(rng.choice([2.0, 3.0, 4.0]))
        base_value = 2.5 / rate + rng.uniform(-0.2, 0.3)  # variație mică
        value = round(base_value, 2)
        trades.append({"give": give, "get": get, "rate": rate, "value": value})
    return trades


def trade_delta(trade: dict) -> dict:
    """Transformă un trade într-un vector delta pe resurse."""
    d = {r: 0.0 for r in RESOURCES}
    d[trade["give"]] -= trade["rate"]
    d[trade["get"]] += 1.0
    return d


# ==============================================================
# 2️⃣ QUBO simplu: recompensă + penalizări pentru conflicte
# ==============================================================

def build_trader_qubo(trades: list[dict], *, conflict_penalty: float = 3.0) -> np.ndarray:
    """
    QUBO simplu:
      E(x) = -∑ value_i x_i + conflict_penalty * ∑ x_i x_j (dacă i,j folosesc același 'give')
    """
    n = len(trades)
    Q = np.zeros((n, n), dtype=float)

    # Recompense pe diagonală (maximizăm valoarea)
    for i, t in enumerate(trades):
        Q[i, i] += -float(t["value"])

    # Penalizări: trade-uri cu același 'give'
    for i in range(n):
        for j in range(i + 1, n):
            if trades[i]["give"] == trades[j]["give"]:
                Q[i, j] += conflict_penalty
                Q[j, i] += conflict_penalty

    return Q


# ==============================================================
# 3️⃣ Afișare clară a trade-urilor
# ==============================================================

def print_trade_list(trades: list[dict]) -> None:
    print_section("Trade-uri generate aleator")
    print(f"{'ID':<3} {'Give':<8} {'Get':<8} {'Rate':<6} {'Value':<6}  {'Delta'}")
    print("-" * 70)
    for i, t in enumerate(trades):
        d = trade_delta(t)
        delta_str = ", ".join(f"{k}:{d[k]:g}" for k in RESOURCES if abs(d[k]) > 1e-9)
        print(f"{i:<3} {t['give']:<8} {t['get']:<8} {t['rate']:<6.1f} {t['value']:<6.2f}  {delta_str}")
    print()


def decode_solution(bitstring: str) -> list[int]:
    return [i for i, b in enumerate(bitstring[::-1]) if b == "1"]


def print_selected_trades(trades: list[dict], bitstring: str) -> None:
    idx = decode_solution(bitstring)
    total_value = sum(trades[i]["value"] for i in idx)
    net = {r: 0.0 for r in RESOURCES}
    for i in idx:
        d = trade_delta(trades[i])
        for r in RESOURCES:
            net[r] += d[r]

    cprint("\n=== Trade-uri selectate ===", Colors.CYAN)
    if not idx:
        print("(niciun trade activ)")
    for i in idx:
        t = trades[i]
        d = trade_delta(t)
        d_str = ", ".join(f"{k}:{d[k]:g}" for k in RESOURCES if abs(d[k]) > 1e-9)
        print(f"#{i}: {t['give']} -> {t['get']} (rate={t['rate']}, value={t['value']}) | Δ {d_str}")

    cprint(f"\nValoare totală (aprox.): {total_value:.2f}", Colors.GREEN)
    cprint("Bilanț resurse:", Colors.YELLOW)
    print("{" + ", ".join(f"{k}: {net[k]:g}" for k in RESOURCES) + "}")


# ==============================================================
# 4️⃣ CLI
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="Quantum Catan — P3 Resource Trader (aleator)")
    parser.add_argument("--trades", type=int, default=8, help="Numărul de trade-uri generate aleator")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="qaoa", choices=["qaoa", "vqe"])
    # QAOA
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="COBYLA")
    parser.add_argument("--shots", type=int, default=256)
    # VQE
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.15)
    # QUBO
    parser.add_argument("--conflict", type=float, default=3.0)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    # 1️⃣ Generează trade-uri aleatoare
    trades = generate_random_trades(args.trades, args.seed)
    print_trade_list(trades)

    # 2️⃣ Construiește QUBO
    cprint("Construim QUBO…", Colors.YELLOW)
    Q = build_trader_qubo(trades, conflict_penalty=args.conflict)
    cprint(f"QUBO matrix {Q.shape} | nnz={np.count_nonzero(Q)}", Colors.CYAN)

    # 3️⃣ Rulează solverul
    if args.method == "qaoa":
        res = run_qaoa_qubo(Q, reps=args.reps, optimizer=args.optimizer, shots=args.shots, seed=args.seed)
    else:
        res = run_vqe_qubo(Q, layers=2, steps=args.steps, lr=args.lr, optimizer="adam")

    bitstring = res["best_bitstring"]
    energy = res["best_energy"]
    cprint(f"\nBest bitstring = {bitstring} | Energy = {energy:.6f}", Colors.GREEN)

    # 4️⃣ Afișează selecția
    print_selected_trades(trades, bitstring)

    # 5️⃣ Ploturi și salvare
    if args.method == "qaoa":
        plot_energy_landscape(res["bitstrings"], Q)
    else:
        eh = res.get("energy_history")
        if eh:
            fig = plt.figure(figsize=(6, 3))
            plt.plot(range(len(eh)), eh)
            plt.title("VQE Energy Convergence")
            plt.xlabel("Iteration"); plt.ylabel("Energy")
            plt.grid(True, alpha=0.3)
            show_or_save(fig, "p3_vqe_convergence.png")

    if args.save:
        out = Path("data"); out.mkdir(exist_ok=True)
        np.save(out / "p3_Q.npy", Q)
        with open(out / "p3_best.txt", "w") as f:
            f.write(f"method={args.method}\nbitstring={bitstring}\nenergy={energy}\n")
        cprint("[Saved] data/p3_Q.npy, data/p3_best.txt, and figures", Colors.GREEN)


if __name__ == "__main__":
    main()
