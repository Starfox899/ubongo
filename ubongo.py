#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 13:39:57 2025

@author: saschah
"""

from dataclasses import dataclass
from typing import FrozenSet, Tuple, List, Set, Optional, Dict, Iterable
from copy import deepcopy
from textwrap import dedent
import random
import hashlib

# ============================================================
# Geometry & Utilities
# ============================================================

Cell = Tuple[int, int]

def normalize(cells: FrozenSet[Cell]) -> FrozenSet[Cell]:
    """Translate the set so min x,y becomes (0,0)."""
    if not cells:
        return cells
    minx = min(x for x, _ in cells)
    miny = min(y for _, y in cells)
    return frozenset((x - minx, y - miny) for x, y in cells)

def bbox(cells: FrozenSet[Cell]) -> Tuple[int, int]:
    """Return width, height of a tight bounding box around the cells."""
    if not cells:
        return (0, 0)
    maxx = max(x for x, _ in cells)
    maxy = max(y for _, y in cells)
    return (maxx + 1, maxy + 1)

def rotate90(cells: FrozenSet[Cell]) -> FrozenSet[Cell]:
    """Rotate 90° CW within local bbox, then normalize."""
    w, h = bbox(cells)
    return normalize(frozenset((y, w - 1 - x) for (x, y) in cells))

def reflect_x(cells: FrozenSet[Cell]) -> FrozenSet[Cell]:
    """Mirror across y-axis within local bbox, then normalize."""
    w, h = bbox(cells)
    return normalize(frozenset((w - 1 - x, y) for (x, y) in cells))

def dihedral_orientations(cells: FrozenSet[Cell]) -> List[FrozenSet[Cell]]:
    """Generate unique orientations under D4 (rotations + reflections)."""
    variants = []
    cur = normalize(cells)
    for _ in range(4):
        variants.append(cur)
        variants.append(reflect_x(cur))
        cur = rotate90(cur)
    uniq: Dict[Tuple[Cell, ...], FrozenSet[Cell]] = {}
    for v in variants:
        key = tuple(sorted(v))
        if key not in uniq:
            uniq[key] = v
    return list(uniq.values())

def neighbors4(c: Cell) -> List[Cell]:
    x, y = c
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def is_connected(cells: FrozenSet[Cell]) -> bool:
    """Check 4-connectivity."""
    if not cells:
        return True
    seen = set()
    stack = [next(iter(cells))]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        for v in neighbors4(u):
            if v in cells and v not in seen:
                stack.append(v)
    return len(seen) == len(cells)

def has_hole(cells: FrozenSet[Cell]) -> bool:
    """
    Detect enclosed voids via flood-fill from outside of an expanded bbox.
    Returns True iff there exists at least one interior void (hole).
    """
    if not cells:
        return False
    minx = min(x for x, _ in cells) - 1
    miny = min(y for _, y in cells) - 1
    maxx = max(x for x, _ in cells) + 1
    maxy = max(y for _, y in cells) + 1
    cell_set = set(cells)
    outside = set()
    stack = [(minx, miny)]

    def in_rect(p: Cell) -> bool:
        return minx <= p[0] <= maxx and miny <= p[1] <= maxy

    while stack:
        u = stack.pop()
        if u in outside or not in_rect(u):
            continue
        outside.add(u)
        for v in neighbors4(u):
            if v not in outside and v not in cell_set:
                stack.append(v)

    for x in range(minx + 1, maxx):
        for y in range(miny + 1, maxy):
            if (x, y) not in cell_set and (x, y) not in outside:
                return True
    return False

def print_shape(cells: FrozenSet[Cell]) -> str:
    """ASCII view using '#' for filled, '.' for empty."""
    if not cells:
        return "(empty)"
    w, h = bbox(cells)
    grid = [['.' for _ in range(w)] for __ in range(h)]
    for x, y in cells:
        grid[y][x] = '#'
    return '\n'.join(''.join(row) for row in grid)


def cells_to_ascii(cells: Iterable[Cell]) -> str:
    """Convert a collection of cells to normalized ASCII art."""
    fs = normalize(frozenset(cells))
    w, h = bbox(fs)
    grid = [['.' for _ in range(w)] for __ in range(h)]
    for x, y in fs:
        grid[y][x] = '#'
    return '\n'.join(''.join(row) for row in grid)


def shape_circumference(cells: FrozenSet[Cell]) -> int:
    """Return number of distinct non-shape neighbors (4-neighborhood)."""
    neighbors: Set[Cell] = set()
    cell_set = set(cells)
    for c in cell_set:
        for n in neighbors4(c):
            if n not in cell_set:
                neighbors.add(n)
    return len(neighbors)


def circumference_ratio(cells: FrozenSet[Cell]) -> float:
    """Return circumference divided by the number of shape cells."""
    if not cells:
        return 0.0
    return shape_circumference(cells) / len(cells)


def puzzle_hash(ascii_art: str) -> int:
    """Return a stable integer hash for the given puzzle shape.

    The hash is computed from the normalized coordinates of all filled
    cells parsed from ``ascii_art`` (``'#'`` denotes filled cells).  Equivalent
    puzzles – i.e. shapes that are mere translations of each other –
    therefore yield identical hashes.
    """

    lines = dedent(ascii_art).splitlines()
    cells: Set[Cell] = set()
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == '#':
                cells.add((x, y))

    normalized = normalize(frozenset(cells))
    serial = ','.join(f"{x}:{y}" for x, y in sorted(normalized))
    digest = hashlib.blake2b(serial.encode('utf-8'), digest_size=16).digest()
    return int.from_bytes(digest, 'big')

# ============================================================
# Polyomino
# ============================================================

@dataclass(frozen=True)
class Polyomino:
    name: str
    base: FrozenSet[Cell]
    orientations: Tuple[FrozenSet[Cell], ...]

def parse_polyomino(name: str, ascii_art: str) -> Polyomino:
    """
    Parse ASCII art with '#' filled cells ('.' or ' ' otherwise).
    Enforces 4-connectivity and 'no holes' invariant for every piece.
    """
    lines = dedent(ascii_art).splitlines()
    cells: Set[Cell] = set()
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == '#':
                cells.add((x, y))
    base = normalize(frozenset(cells))
    assert is_connected(base), f"{name} must be 4-connected"
    assert not has_hole(base), f"{name} must not contain holes"
    orients = tuple(dihedral_orientations(base))
    return Polyomino(name=name, base=base, orientations=orients)

# ============================================================
# Solver (cover all target cells; pieces at most once; required pieces)
# ============================================================

class SolveStats:
    def __init__(self):
        self.nodes = 0
        self.solutions = 0
        # Each element is a list of placements for a solution. A placement is
        # (piece name, orientation cells, offset)
        self.solutions_detail: List[
            List[Tuple[str, FrozenSet[Cell], Tuple[int, int]]]
        ] = []

def solve_cover(
    shape: FrozenSet[Cell],
    pieces: List[Polyomino],
    required: Set[str] = frozenset(),
    limit: int = 1,
    rng: Optional[random.Random] = None,
) -> SolveStats:
    """
    Backtracking exact-cover style:
    - Cover all cells of ``shape`` using the supplied ``pieces`` at most once.
    - Pieces listed in ``required`` must be part of every accepted cover.
    - Search stops after ``limit`` solutions are found (``limit`` > solutions
      allows enumeration of all solutions).

    Returns a :class:`SolveStats` instance containing:
      * ``solutions`` – number of full covers found.
      * ``nodes`` – number of placement attempts explored.
      * ``solutions_detail`` – for each solution a list of placements
        ``(piece name, orientation cells, offset)`` describing how each piece
        was positioned.
    """
    remaining = set(shape)
    stats = SolveStats()
    if rng is None:
        rng = random

    piece_orients = [list(p.orientations) for p in pieces]
    name_to_idx = {p.name: i for i, p in enumerate(pieces)}
    required_idx = {name_to_idx[n] for n in required if n in name_to_idx}

    # Track placements of pieces in the current partial solution; index aligned
    # with ``pieces`` list.
    placements: List[Optional[Tuple[str, FrozenSet[Cell], Tuple[int, int]]]] = [
        None
    ] * len(pieces)

    def recurse(remaining_cells: Set[Cell], used_required: Set[int]):
        if not remaining_cells:
            if required_idx.issubset(used_required):
                stats.solutions += 1
                stats.solutions_detail.append(
                    deepcopy([p for p in placements if p is not None])
                )
            return
        if stats.solutions >= limit:
            return

        # Pivot: top-left remaining cell (simple heuristic)
        pivot = min(remaining_cells, key=lambda c: (c[1], c[0]))

        # Try unused pieces; smaller area first can help
        cand_idxs = [i for i, p in enumerate(pieces) if placements[i] is None]
        cand_idxs.sort(key=lambda i: len(pieces[i].base))

        for i in cand_idxs:
            orients = piece_orients[i]
            rng.shuffle(orients)
            for orient in orients:
                # Place such that some cell of the orient hits pivot
                for (px, py) in orient:
                    ox, oy = pivot[0] - px, pivot[1] - py
                    placed = {(x + ox, y + oy) for (x, y) in orient}
                    if not placed.issubset(remaining_cells):
                        continue
                    stats.nodes += 1
                    placements[i] = (pieces[i].name, orient, (ox, oy))
                    used_req2 = used_required | ({i} if i in required_idx else set())
                    recurse(remaining_cells - placed, used_req2)
                    placements[i] = None
                    if stats.solutions >= limit:
                        return

    recurse(remaining, set())
    return stats

# ============================================================
# Random connected union (hole-free) from a given subset
# ============================================================

def random_connected_union(
    pieces: List[Polyomino],
    max_w: int,
    max_h: int,
    rng: Optional[random.Random] = None,
) -> Optional[Tuple[FrozenSet[Cell], List[Tuple[Polyomino, FrozenSet[Cell], Tuple[int, int]]]]]:
    """
    Place all 'pieces' into a single 4-connected union within (max_w, max_h).
    Rejects any placement step that introduces a hole in the union.
    Returns (normalized_shape, placements) or None if it fails.
    """
    if rng is None:
        rng = random.Random()
    order = pieces[:]
    rng.shuffle(order)

    placements: List[Tuple[Polyomino, FrozenSet[Cell], Tuple[int, int]]] = []
    union: Set[Cell] = set()

    def try_place(i: int) -> bool:
        nonlocal union
        if i == len(order):
            return True

        p = order[i]
        orients = list(p.orientations)
        rng.shuffle(orients)

        if i == 0:
            # First piece at origin
            o = rng.choice(orients)
            placements.append((p, o, (0, 0)))
            u2 = union | set(o)
            shape = normalize(frozenset(u2))
            w, h = bbox(shape)
            if w > max_w or h > max_h or has_hole(shape):
                placements.pop()
                return False
            union = set(u2)
            return try_place(i + 1)

        # Build a border (empty neighbors of current union)
        border: Set[Cell] = set()
        for c in union:
            for n in neighbors4(c):
                if n not in union:
                    border.add(n)

        candidates: List[Tuple[FrozenSet[Cell], Tuple[int, int], Set[Cell]]] = []
        for o in orients:
            o_cells = list(o)
            for pc in o_cells:
                for b in border:
                    ox, oy = b[0] - pc[0], b[1] - pc[1]
                    placed = {(x + ox, y + oy) for (x, y) in o_cells}
                    if placed & union:
                        continue
                    # Must be 4-adjacent to union
                    touches = any(n in union for c in placed for n in neighbors4(c))
                    if not touches:
                        continue
                    u2 = union | placed
                    shape = normalize(frozenset(u2))
                    w, h = bbox(shape)
                    if (max_w and w > max_w) or (max_h and h > max_h):
                        continue
                    if has_hole(shape):
                        continue
                    candidates.append((o, (ox, oy), placed))

        rng.shuffle(candidates)
        for o, offset, placed in candidates:
            placements.append((p, o, offset))
            # mutate union
            for c in placed:
                union.add(c)
            if try_place(i + 1):
                return True
            # backtrack
            for c in placed:
                union.remove(c)
            placements.pop()

        return False

    ok = try_place(0)
    if not ok:
        return None

    u_norm = normalize(frozenset(union))
    # Normalize placement offsets to match normalized union
    minx = min(x for x, _ in union)
    miny = min(y for _, y in union)
    placements = [(p, o, (ox - minx, oy - miny)) for (p, o, (ox, oy)) in placements]
    return u_norm, placements

# ============================================================
# Tiering
# ============================================================

def tier_from_nodes(nodes: int, pieces_count: int, has_holes_flag: bool) -> str:
    if has_holes_flag or pieces_count >= 6 or nodes > 10000:
        return "HARD"
    if nodes > 800 or pieces_count >= 5:
        return "MEDIUM"
    return "EASY"

# ============================================================
# High-level Generator (MANDATORY piece is a PARAMETER)
# ============================================================

def generate_puzzle_with_mandatory_alt(
    library: List[Polyomino],
    k: int,
    max_w: int,
    max_h: int,
    mandatory_piece: str,            # <-- configurable mandatory piece name
    seed: Optional[int] = None,
    max_attempts: int = 400
) -> Optional[dict]:
    """
    Generate a puzzle where:
      1) A constructive solution exists using k pieces sampled from 'library' EXCLUDING 'mandatory_piece'.
      2) An alternative solution exists using pieces from 'library' that MUST include 'mandatory_piece'.
    Multiple solutions are allowed; uniqueness is NOT enforced.

    Returns a dict with metadata (ascii, tier, nodes, etc.) or None if not found in attempts.
    """
    rng = random.Random(seed)
    names = [p.name for p in library]
    assert mandatory_piece in names, f"Library must contain required piece '{mandatory_piece}'"

    for attempt in range(1, max_attempts + 1):
        # Sample constructive subset S without the mandatory piece
        pool_wo_mandatory = [p for p in library if p.name != mandatory_piece]
        if len(pool_wo_mandatory) < k:
            raise ValueError("Not enough non-mandatory pieces to sample from.")

        subset = rng.sample(pool_wo_mandatory, k)
        layout = random_connected_union(subset, max_w=max_w, max_h=max_h, rng=rng)
        if layout is None:
            continue
        shape, placements = layout

        # Sanity: area match (guarantees constructive solvability by construction)
        if len(shape) != sum(len(p.base) for p in subset):
            continue

        # Solve once with subset S (measure difficulty)
        stats_S = solve_cover(shape, subset, required=set(), limit=1, rng=rng)
        if stats_S.solutions < 1:
            # Should not happen given constructive build, but keep guard
            continue

        # Verify the existence of an alternative solution that MUST include the mandatory piece
        stats_alt = solve_cover(shape, library, required={mandatory_piece}, limit=1, rng=rng)
        if stats_alt.solutions < 1:
            # Discard puzzles that do not admit a solution including the mandatory piece
            continue

        # Final checks: target hole-free invariant (should always be true here)
        shape_hole = has_hole(shape)
        if shape_hole:
            continue

        tier = tier_from_nodes(stats_S.nodes, k, shape_hole)
        circ = shape_circumference(shape)
        ratio = circumference_ratio(shape)

        return {
            "shape": shape,
            "size": bbox(shape),
            "circumference": circ,
            "circumference_ratio": ratio,
            "subset_S": [p.name for p in subset],      # constructive set (no mandatory piece)
            "solutions_with_S": stats_S.solutions,                # >= 1
            "nodes_with_S": stats_S.nodes,
            "has_mandatory_alternative": True,
            "alternative_solutions": stats_alt.solutions_detail,
            "nodes_with_mandatory_alt": stats_alt.nodes,
            "tier": tier,
            "attempts": attempt,
            "ascii": print_shape(shape),
            # Optional: placements from the constructive build (offsets in normalized frame)
            "placements_S": [(p.name, off) for (p, _, off) in placements],
        }

    return None

# ============================================================
# Example Library & CLI Usage
# ============================================================

PIECES: List[Polyomino] = [
    # Mandatory piece (data name can be "I3"; we do NOT hardcode it anywhere else)
    parse_polyomino("I3", "###"),
    # More pieces (hole-free by design)
    parse_polyomino("T4", "###\n.#."),
    parse_polyomino("L4", "#..\n#..\n##."),
    parse_polyomino("S4", ".##\n##."),
    parse_polyomino("T5", "####\n.#.."),
    parse_polyomino("L5", "#..\n#..\n#..\n##."),
    parse_polyomino("P5", "##\n##\n#."),
    # Add more as needed:
    # parse_polyomino("Z4", "##.\n.##"),
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate an Ubongo puzzle")
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducible puzzle generation"
    )
    parser.add_argument(
        "--pieces",
        type=int,
        default=3,
        help="Number of pieces to construct the puzzle with",
    )
    args = parser.parse_args()

    # Example: Build a puzzle that has a constructive solution with k pieces
    # (excluding the mandatory one), and also admits an alternative solution that
    # must include the mandatory piece "I3".
    puzzle = generate_puzzle_with_mandatory_alt(
        library=PIECES,
        k=args.pieces,
        max_w=8,
        max_h=6,
        mandatory_piece="I3",   # configurable at call-site
        seed=args.seed,
        max_attempts=600,
    )

    if puzzle:
        print(
            f"Tier: {puzzle['tier']} | size: {puzzle['size']} | attempts: {puzzle['attempts']}"
        )
        print(
            f"Circumference: {puzzle['circumference']} | "
            f"Circumference ratio: {puzzle['circumference_ratio']:.2f}"
        )
        print("Constructive subset (no mandatory piece):", puzzle["subset_S"])
        print(
            "Alternative solutions (w/ mandatory piece):",
            len(puzzle["alternative_solutions"]),
        )
        print(
            "1st alternative solution (w/ mandatory piece):",
            [s for s, _, _ in puzzle["alternative_solutions"][0]],
        )
        print("ASCII target shape:\n" + puzzle["ascii"])
    else:
        print(
            "No puzzle found within attempts. Consider increasing attempts, size limits, or library size."
        )
