from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Set
import math
import random
from collections import defaultdict

# -----------------------------
# Dice helpers (2d6)
# -----------------------------
DICE_PROB: Dict[int, float] = {
    2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
    7: 6/36, 8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36
}
def dice_prob(num: int) -> float:
    return DICE_PROB.get(num, 0.0)

# -----------------------------
# Axial hex grid (pointy-top)
# -----------------------------
AXIAL_DIRS = [(+1,0),(+1,-1),(0,-1),(-1,0),(-1,+1),(0,+1)]

def axial_ring(radius: int) -> List[Tuple[int,int]]:
    q, r = 0, -radius
    coords = []
    if radius == 0:
        return [(0,0)]
    for dir_idx in range(6):
        for _ in range(radius):
            coords.append((q, r))
            dq, dr = AXIAL_DIRS[dir_idx]
            q += dq; r += dr
    return coords

def axial_disk(radius: int) -> List[Tuple[int,int]]:
    coords = [(0,0)]
    for k in range(1, radius+1):
        coords.extend(axial_ring(k))
    return coords

def hex_corners(center: Tuple[float,float], size: float=1.0) -> List[Tuple[float,float]]:
    cx, cy = center
    corners = []
    for i in range(6):
        angle = math.radians(60 * i - 30)  # pointy-top
        corners.append((cx + size * math.cos(angle), cy + size * math.sin(angle)))
    return corners

def axial_to_xy(q: int, r: int, size: float=1.0) -> Tuple[float,float]:
    x = size * (math.sqrt(3) * (q + r/2))
    y = size * (3/2 * r)
    return (x, y)

# -----------------------------
# Board data structures
# -----------------------------
RESOURCES_CLASSIC = ["wood","brick","sheep","wheat","ore"]

@dataclass(frozen=True)
class Tile:
    id: int
    axial: Tuple[int,int]
    resource: str
    number: Optional[int] = None  # 2..12, nu 7

@dataclass(frozen=True)
class Node:
    id: int
    xy: Tuple[float,float]

@dataclass(frozen=True)
class Edge:
    id: int
    u: int
    v: int

@dataclass
class Board:
    radius: int = 2
    tiles: List[Tile] = field(default_factory=list)
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    # adjacency
    node_neighbors: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    node_tiles: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    tile_nodes: Dict[int, List[int]] = field(default_factory=dict)
    edge_index: Dict[Tuple[int,int], int] = field(default_factory=dict)  # (min(u,v),max(u,v))->edge id

    @property
    def n_nodes(self) -> int: return len(self.nodes)
    @property
    def n_edges(self) -> int: return len(self.edges)
    @property
    def n_tiles(self) -> int: return len(self.tiles)

    # ---- Selectori pentru bugetul de qubiți ----
    def settlement_candidates(self, n_qubits: int) -> Tuple[List[int], Dict[int,int]]:
        """Alege până la n_qubits intersecții (noduri) cu scor mare: suma P(2d6=number) pentru tile-urile adiacente."""
        scores = []
        for node in self.nodes:
            weight = 0.0
            for t_idx in self.node_tiles[node.id]:
                num = self.tiles[t_idx].number
                weight += dice_prob(num) if num is not None else 0.0
            scores.append((weight, node.id))
        scores.sort(reverse=True)
        selected = [nid for _, nid in scores[:n_qubits]]
        var_map = {i: nid for i, nid in enumerate(selected)}
        return selected, var_map

    def road_candidates(self, n_qubits: int) -> Tuple[List[int], Dict[int,int]]:
        """Alege până la n_qubits muchii (drumuri) scorate prin suma scorurilor celor două capete."""
        node_w = {}
        for node in self.nodes:
            w = 0.0
            for t_idx in self.node_tiles[node.id]:
                num = self.tiles[t_idx].number
                w += dice_prob(num) if num is not None else 0.0
            node_w[node.id] = w
        scores = []
        for e in self.edges:
            w = node_w[e.u] + node_w[e.v]
            scores.append((w, e.id))
        scores.sort(reverse=True)
        selected = [eid for _, eid in scores[:n_qubits]]
        var_map = {i: eid for i, eid in enumerate(selected)}
        return selected, var_map

    def incidence_matrix(self, use_selected_edges: Optional[Iterable[int]]=None):
        """Returnează matricea de incidență (E x V) din {−1,0,1} pentru formulări de QUBO/ILP."""
        V = self.n_nodes
        edges = self.edges if use_selected_edges is None else [e for e in self.edges if e.id in set(use_selected_edges)]
        mat = []
        for e in edges:
            row = [0]*V
            row[e.u] = +1
            row[e.v] = -1
            mat.append(row)
        return mat

# -----------------------------
# Generare tablă clasică (raza 2)
# -----------------------------
def generate_classic_board(seed: Optional[int]=None) -> Board:
    """Radius-2 (19 tile-uri). Distribuție simplificată de resurse + token-uri (fără 7). Determinist cu seed."""
    rng = random.Random(seed)
    radius = 2
    hexes = axial_disk(radius)      # 19 coordonate

    # approx. distribuția Catan + Desert
    resources_pool = ["wood"]*4 + ["brick"]*3 + ["sheep"]*4 + ["wheat"]*4 + ["ore"]*3 + ["desert"]
    rng.shuffle(resources_pool)
    numbers_pool = [2,3,3,4,4,5,5,6,6,8,8,9,9,10,10,11,11,12]  # 18 numere, fără 7
    rng.shuffle(numbers_pool)

    tiles: List[Tile] = []
    num_iter = iter(numbers_pool)
    for tidx, (q,r) in enumerate(hexes):
        res = resources_pool[tidx]
        num = None if res == "desert" else next(num_iter)
        tiles.append(Tile(id=tidx, axial=(q,r), resource=res, number=num))

    # Construcția nodurilor/muchiilor din colțurile hexurilor
    node_id_by_pos: Dict[Tuple[int,int], int] = {}
    nodes: List[Node] = []
    edges: List[Edge] = []
    node_tiles: Dict[int, Set[int]] = defaultdict(set)
    node_neighbors: Dict[int, Set[int]] = defaultdict(set)
    tile_nodes: Dict[int, List[int]] = {}

    SCALE = 1000  # pentru deduplicare prin cuantizare
    def qpos(p: Tuple[float,float]) -> Tuple[int,int]:
        return (int(round(p[0]*SCALE)), int(round(p[1]*SCALE)))

    for t in tiles:
        cx, cy = axial_to_xy(*t.axial, size=1.0)
        corners = hex_corners((cx,cy), size=1.0)
        corner_ids = []
        for c in corners:
            qp = qpos(c)
            if qp not in node_id_by_pos:
                nid = len(nodes)
                node_id_by_pos[qp] = nid
                nodes.append(Node(id=nid, xy=c))
            else:
                nid = node_id_by_pos[qp]
            node_tiles[nid].add(t.id)
            corner_ids.append(nid)
        tile_nodes[t.id] = corner_ids
        for i in range(6):
            u = corner_ids[i]
            v = corner_ids[(i+1)%6]
            if v not in node_neighbors[u]:
                node_neighbors[u].add(v)
                node_neighbors[v].add(u)

    edge_index: Dict[Tuple[int,int], int] = {}
    for u, neighs in node_neighbors.items():
        for v in neighs:
            if u < v:
                eid = len(edges)
                edges.append(Edge(id=eid, u=u, v=v))
                edge_index[(u,v)] = eid

    return Board(
        radius=radius,
        tiles=tiles,
        nodes=nodes,
        edges=edges,
        node_neighbors=node_neighbors,
        node_tiles=node_tiles,
        tile_nodes=tile_nodes,
        edge_index=edge_index
    )

# -----------------------------
# Utilitar pentru „număr de qubiți”
# -----------------------------
def board_for_qubits(n_qubits: int, mode: str="settlement", seed: Optional[int]=None) -> Tuple[Board, List[int], Dict[int,int]]:
    """Generează tabla și preselectează variabilele ca să încapă în n_qubits.
    mode: 'settlement' (intersecții) sau 'road' (drumuri).
    Returnează (board, selected_ids, var_map) unde var_map: index_variabilă → id original.
    """
    b = generate_classic_board(seed=seed)
    if mode == "settlement":
        selected, var_map = b.settlement_candidates(n_qubits)
    elif mode == "road":
        selected, var_map = b.road_candidates(n_qubits)
    else:
        raise ValueError("mode must be 'settlement' sau 'road'")
    return b, selected, var_map
