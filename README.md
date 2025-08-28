# Ubongo Puzzle Toolkit

Ubongo is a fast-paced tile‑placement game where players race to cover a given board with a set of polyomino pieces.  This repository provides a small Python toolkit for experimenting with Ubongo puzzles.  It includes utilities for defining pieces, generating puzzle boards and verifying solutions.

## Code Intention

The project focuses on representing **polyomino** pieces and using them to construct and solve exact‑cover puzzles.  Key capabilities include:

* Parsing ASCII art into polyomino definitions.
* Enumerating all unique rotations and reflections of each piece.
* Backtracking search to cover a target shape with selected pieces (optionally requiring certain pieces).
* Helper functions for building random hole‑free boards and sampling piece sets.

The code is intended as a playground for exploring Ubongo‑style puzzles rather than a full game implementation.

## Data Model

The core data structures model pieces and boards on a square grid:

### Cells

A **Cell** is a pair of integer coordinates `(x, y)` representing a unit square on the grid.

```python
Cell = Tuple[int, int]
```

### Shapes

A **shape** (board region or piece footprint) is stored as a `FrozenSet[Cell]`.  Normalisation functions translate shapes so that the smallest `x` and `y` coordinate becomes `(0, 0)`.  This representation makes shapes easy to compare, transform and overlay.

### Polyomino

Each puzzle piece is represented by the `Polyomino` dataclass:

```python
@dataclass(frozen=True)
class Polyomino:
    name: str
    base: FrozenSet[Cell]
    orientations: Tuple[FrozenSet[Cell], ...]
```

* `name` – identifier used in solutions and tests.
* `base` – canonical cell set of the piece, normalised to the origin.
* `orientations` – all unique rotations and mirror images of the piece, pre‑computed under the dihedral group so the solver can iterate efficiently.

Pieces are typically defined via `parse_polyomino(name, ascii_art)` where `ascii_art` uses `#` for filled cells and `.` or space for empty cells.

## Solver Overview

The solver uses a depth‑first exact‑cover search.  Given a target shape and a collection of `Polyomino` pieces it tries to place each piece at most once so that every cell of the shape is covered with no overlaps.  Optional constraints enforce that specific pieces must appear in every solution.  Search statistics such as node count and the list of piece placements are collected in a `SolveStats` object.

## Example

```python
from ubongo import parse_polyomino, solve_cover

shape = frozenset({(0,0), (1,0), (0,1), (1,1)})  # 2×2 square
p1 = parse_polyomino("I2", "##")
p2 = parse_polyomino("L2", "#\n#")

stats = solve_cover(shape, [p1, p2])
print(stats.solutions)  # -> 2
```

This demonstrates how pieces are defined and used to cover a simple shape.

## Command-line puzzle generation

Running `python ubongo.py` will generate a sample puzzle using the built-in
piece library.  You can supply a `--seed` argument to make puzzle generation
reproducible:

```bash
python ubongo.py --seed 42
```

Using the same seed will always yield the same puzzle layout and statistics.

## Circumference ratio experiment

The repository includes `experiment.py` which samples 100 random puzzles with
different seeds and plots a histogram of their circumference-to-area ratios.
Run:

```bash
python experiment.py
```

`matplotlib` is required to display the histogram.

---

This README is a draft and can be extended with more detailed usage examples and contributor guidelines.

