from __future__ import annotations
import os
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Iterable, List

import os, matplotlib, matplotlib.pyplot as plt
from pathlib import Path

def show_or_save(fig, fname: str | None = None, folder: str = "data"):
    """
    If running without a GUI backend (Agg) or SAVE_PLOTS=1, save to PNG.
    Otherwise, plt.show().
    """
    backend = matplotlib.get_backend().lower()
    save = backend.endswith("agg") or os.environ.get("SAVE_PLOTS", "1") == "1"
    if save and fname:
        Path(folder).mkdir(exist_ok=True)
        path = Path(folder) / fname
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print(f"[viz] saved {path}")
        plt.close(fig)
    else:
        plt.show()

# ===============================
# Random / Seed utils
# ===============================

_GLOBAL_SEED: Optional[int] = None

def set_seed(seed: Optional[int] = None) -> int:
    """Setează seed global pentru reproducibilitate (random + numpy). Returnează seed-ul efectiv folosit."""
    global _GLOBAL_SEED
    if seed is None:
        seed = int(time.time() * 1000) % (2**32 - 1)
    _GLOBAL_SEED = seed
    random.seed(seed)
    np.random.seed(seed)
    return seed

def get_seed() -> Optional[int]:
    """Returnează seed-ul curent, dacă a fost setat."""
    return _GLOBAL_SEED


# ===============================
# Timing helpers
# ===============================

@dataclass
class Timer:
    """Context manager pentru măsurarea duratei unui bloc de cod."""
    name: str = ""
    silent: bool = False
    start_time: float = 0.0
    elapsed: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.elapsed = time.perf_counter() - self.start_time
        if not self.silent:
            print(f"⏱️  {self.name or 'Timer'}: {self.elapsed:.3f} s")

def now_str(fmt: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """Returnează timestamp curent ca string, pentru fișiere/loguri."""
    return time.strftime(fmt, time.localtime())


# ===============================
# CLI / notebook argument parsing
# ===============================

def get_env_int(name: str, default: int) -> int:
    """Citește o valoare întreagă din environment, ex: NUM_QUBITS=24."""
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def get_env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


# ===============================
# Logging & pretty print
# ===============================

class Colors:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"

def cprint(text: str, color: str = Colors.GREEN, bold: bool = False) -> None:
    """Print colorat în terminal."""
    style = color
    if bold:
        style += Colors.BOLD
    print(f"{style}{text}{Colors.END}")

def print_section(title: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}=== {title} ==={Colors.END}")

def print_dict(d: Dict[Any, Any], title: Optional[str] = None):
    if title:
        print_section(title)
    for k, v in d.items():
        print(f"  {k:<20} : {v}")


# ===============================
# Data conversion helpers
# ===============================

def normalize(v: Iterable[float]) -> np.ndarray:
    """Normalizează vectorul v la sumă 1."""
    arr = np.array(v, dtype=float)
    s = np.sum(arr)
    return arr / s if s != 0 else arr

def argmax_dict(d: Dict[Any, float]) -> Tuple[Any, float]:
    """Returnează cheia cu valoarea maximă."""
    if not d:
        return None, float("nan")
    key = max(d, key=d.get)
    return key, d[key]

def topk_dict(d: Dict[Any, float], k: int) -> List[Tuple[Any, float]]:
    """Returnează primii k itemi ordonați descrescător."""
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:k]


# ===============================
# Math helpers
# ===============================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def softmax(xs: Iterable[float]) -> np.ndarray:
    arr = np.array(xs, dtype=float)
    exps = np.exp(arr - np.max(arr))
    return exps / np.sum(exps)


# ===============================
# File utilities
# ===============================

def ensure_dir(path: str):
    """Creează directorul dacă nu există."""
    if not os.path.exists(path):
        os.makedirs(path)

def unique_filename(prefix: str = "out", ext: str = ".txt") -> str:
    """Returnează un nume unic bazat pe timp."""
    return f"{prefix}_{now_str()}{ext}"


# ===============================
# Example usage
# ===============================
if __name__ == "__main__":
    seed = set_seed(42)
    print_section("Test utils.py")
    cprint(f"Seed global: {seed}", Colors.BLUE)
    with Timer("sleep"):
        time.sleep(0.5)
    arr = [0.1, 0.3, 0.6]
    print_dict({"softmax": softmax(arr).tolist()}, "Vector test")
