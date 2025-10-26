from __future__ import annotations

from typing import Iterable, Optional, Tuple, Dict, List
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

# Dependențe interne
from catan.board import Board, axial_to_xy, hex_corners, dice_prob


# -----------------------------
# Palete & utilitare
# -----------------------------
RESOURCE_COLOR: Dict[str, Tuple[float, float, float]] = {
    "wood":  (0.30, 0.62, 0.36),  # verde
    "brick": (0.75, 0.33, 0.24),  # cărămidă
    "sheep": (0.55, 0.78, 0.32),  # lime
    "wheat": (0.93, 0.80, 0.30),  # galben
    "ore":   (0.45, 0.48, 0.52),  # gri
    "desert":(0.91, 0.84, 0.64),  # nisip
}

def resource_color(res: str) -> Tuple[float, float, float]:
    return RESOURCE_COLOR.get(res, (0.8, 0.8, 0.8))

def number_face_color(n: Optional[int]) -> Tuple[float, float, float]:
    # 6 și 8 evidențiate
    if n in (6, 8):
        return (0.85, 0.25, 0.25)
    return (0.15, 0.15, 0.15)

def _auto_bounds(board: Board, pad: float = 1.5) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for t in board.tiles:
        cx, cy = axial_to_xy(*t.axial, size=1.0)
        xs.append(cx); ys.append(cy)
    if not xs:
        return (-5, 5, -5, 5)
    return (min(xs)-pad, max(xs)+pad, min(ys)-pad, max(ys)+pad)


# -----------------------------
# Plot tabla Catan
# -----------------------------
def plot_board(
    board: Board,
    ax: Optional[plt.Axes] = None,
    *,
    show_numbers: bool = True,
    show_node_ids: bool = False,
    show_edge_ids: bool = False,
    tile_alpha: float = 0.90,
    edge_alpha: float = 0.50,
):
    """Desenează hex-urile, muchiile și (opțional) ID-urile nodurilor/muchiilor."""
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        created_fig = True

    # 1) tiles
    for t in board.tiles:
        cx, cy = axial_to_xy(*t.axial, size=1.0)
        corners = hex_corners((cx, cy), size=1.0)
        poly = Polygon(corners, closed=True, facecolor=resource_color(t.resource),
                       edgecolor=(0.15, 0.15, 0.15), linewidth=1.5, alpha=tile_alpha)
        ax.add_patch(poly)

        # number token
        if show_numbers and t.number is not None:
            circ = Circle((cx, cy), radius=0.42, facecolor="white",
                          edgecolor=(0.10, 0.10, 0.10), linewidth=1.2, zorder=5)
            ax.add_patch(circ)
            ax.text(cx, cy, f"{t.number}\n{dice_prob(t.number):.2f}",
                    ha="center", va="center",
                    fontsize=10, color=number_face_color(t.number), zorder=6)

    # 2) edges (linii între noduri)
    for e in board.edges:
        u = board.nodes[e.u].xy
        v = board.nodes[e.v].xy
        ax.plot([u[0], v[0]], [u[1], v[1]], linewidth=1.8, color=(0.05, 0.05, 0.05), alpha=edge_alpha)
        if show_edge_ids:
            mx = (u[0] + v[0]) / 2.0
            my = (u[1] + v[1]) / 2.0
            ax.text(mx, my, str(e.id), fontsize=7, color=(0.15, 0.15, 0.15))

    # 3) nodes (intersecții)
    xs = [n.xy[0] for n in board.nodes]
    ys = [n.xy[1] for n in board.nodes]
    ax.scatter(xs, ys, s=18, c="black", zorder=7)
    if show_node_ids:
        for n in board.nodes:
            ax.text(n.xy[0] + 0.05, n.xy[1] + 0.05, str(n.id), fontsize=7, color="black")

    # 4) ax styling
    xmin, xmax, ymin, ymax = _auto_bounds(board)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if created_fig:
        return fig, ax
    return ax


# -----------------------------
# Suprapuneri: selecții / soluții
# -----------------------------
def highlight_nodes(
    board,
    node_ids,
    ax=None,
    color=(0.25, 0.60, 1.00),
    size: float = 80,
    lw: float = 1.8,
    label: str = None,
    zorder: float = 9.0,
    **kwargs,
):
    """
    Highlight a set of node IDs on the board.

    Works with boards that expose either:
      - board.node_positions: dict {node_id: (x, y)}
      - board.nodes: list/seq with items having .id and .xy
    Accepts both `size` and `s` (back-compat).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # 1) Build an id -> (x, y) map robustly
    pos = {}
    node_pos_dict = getattr(board, "node_positions", None)
    if isinstance(node_pos_dict, dict) and node_pos_dict:
        # Already a dict of coordinates
        pos = {int(k): tuple(v) for k, v in node_pos_dict.items()}
    else:
        # Derive from board.nodes[*].id / .xy
        nodes_seq = getattr(board, "nodes", [])
        for n in nodes_seq:
            nid = getattr(n, "id", None)
            xy  = getattr(n, "xy", None)
            if nid is not None and xy is not None:
                pos[int(nid)] = (float(xy[0]), float(xy[1]))

    # 2) Collect points to draw
    xs, ys = [], []
    for nid in node_ids:
        nid = int(nid)
        if nid in pos:
            x, y = pos[nid]
            xs.append(x)
            ys.append(y)

    # 3) Merge size parameters cleanly
    scatter_size = kwargs.pop("s", size)

    # 4) Draw hollow circles above the black node dots
    ax.scatter(
        xs, ys,
        facecolors="none",
        edgecolors=color,
        linewidths=lw,
        s=scatter_size,
        label=label,
        zorder=zorder,
        **kwargs,
    )
    return ax


def highlight_edges(
    board: Board,
    edges: Iterable[int],
    ax: Optional[plt.Axes] = None,
    *,
    color=(0.10, 0.55, 0.95),
    width: float = 3.5,
    label: Optional[str] = None,
):
    """Evidențiază un subset de muchii (ex.: o soluție de 'longest road' sau selecția de qubiți)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
        plot_board(board, ax=ax)

    for eid in edges:
        e = board.edges[eid]
        u = board.nodes[e.u].xy
        v = board.nodes[e.v].xy
        ax.plot([u[0], v[0]], [u[1], v[1]], linewidth=width, color=color, zorder=8)
    if label:
        ax.legend(loc="upper right")
    return ax


# -----------------------------
# Helpers „one-liners”
# -----------------------------
def plot_settlement_selection(board: Board, selected_nodes: Iterable[int], title: str = "Settlement variables"):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_board(board, ax=ax)
    highlight_nodes(board, selected_nodes, ax=ax, label=f"{len(list(selected_nodes))} nodes")
    ax.set_title(title)
    return fig, ax


def plot_road_selection(board: Board, selected_edges: Iterable[int], title: str = "Road variables"):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_board(board, ax=ax)
    highlight_edges(board, selected_edges, ax=ax, label=f"{len(list(selected_edges))} edges")
    ax.set_title(title)
    return fig, ax


def save_board_png(board: Board, path: str, **kwargs):
    """Shortcut pentru a salva rapid o reprezentare a tablei."""
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_board(board, ax=ax, **kwargs)
    fig.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(fig)
