from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class QuboModel:
    Q: np.ndarray
    labels: List[str]
    idx_of: Dict[int, int]
    id_of: Dict[int, int]


# ----------------------------- Helpers -----------------------------

def _build_index_maps(ids: Sequence[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    idx_of = {nid: i for i, nid in enumerate(ids)}
    id_of = {i: nid for i, nid in enumerate(ids)}
    return idx_of, id_of


def _edge_endpoints(edge: Any) -> Optional[Tuple[int, int]]:
    if isinstance(edge, (tuple, list)) and len(edge) == 2:
        return int(edge[0]), int(edge[1])
    for a, b in [
        ("u", "v"), ("a", "b"), ("i", "j"),
        ("n1", "n2"), ("start", "end"),
        ("src", "dst"), ("node_u", "node_v"),
    ]:
        if hasattr(edge, a) and hasattr(edge, b):
            return int(getattr(edge, a)), int(getattr(edge, b))
    for attr in ("nodes", "endpoints", "vertices"):
        if hasattr(edge, attr):
            val = getattr(edge, attr)
            val = val() if callable(val) else val
            if isinstance(val, (tuple, list)) and len(val) == 2:
                return int(val[0]), int(val[1])
    return None


def _edge_endpoints_on_board(board: Any, e: Any) -> Optional[Tuple[int, int]]:
    ep = _edge_endpoints(e)
    if ep is not None:
        return ep
    if isinstance(e, int):
        edges = getattr(board, "edges", [])
        if isinstance(edges, (list, tuple)) and 0 <= e < len(edges):
            return _edge_endpoints(edges[e])
    for attr in ("edge_to_nodes", "edge_endpoints", "edge_index_to_nodes"):
        mapping = getattr(board, attr, None)
        if mapping is None:
            continue
        val = mapping(e) if callable(mapping) else (mapping.get(e) if isinstance(mapping, dict) else None)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            return int(val[0]), int(val[1])
    return None


# ========================= P1: Settlements ==========================

def _weights_settlement(board: Any, selected_nodes: Sequence[int]) -> np.ndarray:
    node_to_tiles = getattr(board, "node_to_tiles", {})
    tile_number   = getattr(board, "tile_number", {})
    dice_prob     = getattr(board, "dice_prob", {})
    w = []
    for nid in selected_nodes:
        s = 0.0
        for tid in node_to_tiles.get(nid, []):
            num = tile_number.get(tid, None)
            if num is None:
                continue
            s += float(dice_prob.get(num, 0.0))
        w.append(s)
    return np.array(w, dtype=float)


def build_settlement_qubo(
    board: Any,
    selected_nodes: Sequence[int],
    *,
    A: Optional[float] = None,
    maximize_weight: bool = True,
    k_exact: Optional[int] = None,     # set to 2 for base-game rule
    mu_exact: Optional[float] = None,  # equality penalty strength
) -> QuboModel:
    """
    Settlement placement QUBO:
      - distance-2 constraint via adjacency penalties
      - optional reward (expected yield)
      - optional exact-k via mu * (sum x - k)^2

    Implementation detail: we fill only the *upper triangle* (i==j and i<j),
    so energy is computed as x^T Q x without double-counting.
    """
    n = len(selected_nodes)
    if n == 0:
        raise ValueError("selected_nodes is empty")

    w = _weights_settlement(board, selected_nodes)
    idx_of, id_of = _build_index_maps(selected_nodes)

    wsum = float(np.sum(w)) if n > 0 else 1.0
    if A is None:
        A = 1.05 * wsum if maximize_weight else 1.0

    # Strong default equality penalty to reliably force EXACT=2
    if k_exact is not None and mu_exact is None:
        mu_exact = 12.0 * (A + max(wsum, 1.0))

    Q = np.zeros((n, n), dtype=float)

    # Reward (diagonal only)
    if maximize_weight:
        for i in range(n):
            Q[i, i] += -w[i]

    # Adjacency (i<j only)
    adj_pairs = set()
    for e in getattr(board, "edges", []):
        ep = _edge_endpoints_on_board(board, e)
        if ep is None:
            continue
        u, v = ep
        if u in idx_of and v in idx_of:
            i, j = idx_of[u], idx_of[v]
            if i > j:
                i, j = j, i
            adj_pairs.add((i, j))
    node_neighbors = getattr(board, "node_neighbors", {})
    for nid in selected_nodes:
        i = idx_of[nid]
        for nb in node_neighbors.get(nid, []):
            if nb in idx_of:
                j = idx_of[nb]
                if i > j:
                    i, j = j, i
                adj_pairs.add((i, j))
    for (i, j) in adj_pairs:
        Q[i, j] += A  # i<j only

    # Exact-k: mu * (sum x - k)^2
    if k_exact is not None and mu_exact is not None and k_exact >= 0:
        # diagonal
        diag_add = mu_exact * (1 - 2 * k_exact)
        for i in range(n):
            Q[i, i] += diag_add
        # off-diagonal (i<j only)
        off = 2.0 * mu_exact
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] += off

    labels = [f"node:{id_of[i]}" for i in range(n)]
    return QuboModel(Q=Q, labels=labels, idx_of=idx_of, id_of=id_of)


__all__ = [
    "QuboModel",
    "_edge_endpoints",
    "_edge_endpoints_on_board",
    "_weights_settlement",
    "build_settlement_qubo",
]
