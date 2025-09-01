#!/usr/bin/env python3
"""Generate a LaTeX document with Ubongo puzzles.

This script reads a JSON file containing a list of puzzle dictionaries as
produced by :func:`generate_puzzle_with_mandatory_alt` in ``ubongo.py`` and
creates a LaTeX document ready for printing.  Each puzzle is drawn using TikZ
with squares sized at 12.5mm so real pieces fit exactly on the printout.

Duplicate puzzles (same layout) are removed using ``puzzle_hash``.  Every puzzle
is annotated with the number of pieces in the constructive subset and its hash
value for reference.

Example usage:
    python print_puzzles.py puzzles.json output.tex
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict, Tuple

from ubongo import puzzle_hash

CELL_MM = 13.0  # millimetres per cell when rendered


def ascii_to_tikz(ascii_art: str) -> str:
    """Return a TikZ picture representing the ASCII shape."""
    lines = [line.rstrip('\n') for line in ascii_art.splitlines() if line.strip('\n')]
    if not lines:
        return ""
    h = len(lines)
    w = max(len(line) for line in lines)
    tikz = [f"\\begin{{tikzpicture}}[x={CELL_MM}mm,y={CELL_MM}mm]"]
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch == '#':
                yy = h - y - 1
                tikz.append(
                    f"\\filldraw[fill=gray!15, draw=black, thick] ({x},{yy}) rectangle ({x+1},{yy+1});"
                )
    tikz.append("\\end{tikzpicture}")
    return "\n".join(tikz)


def unique_puzzles(puzzles: Iterable[Dict]) -> List[Tuple[int, Dict]]:
    """Return a list of (hash, puzzle) for unique puzzles."""
    seen = set()
    uniq: List[Tuple[int, Dict]] = []
    for p in puzzles:
        h = puzzle_hash(p["ascii"])
        if h in seen:
            continue
        seen.add(h)
        uniq.append((h, p))
    return uniq


def puzzles_to_latex(puzzles: Iterable[Dict]) -> str:
    """Convert puzzles to a LaTeX document string."""
    entries: List[str] = []
    for h, p in unique_puzzles(puzzles):
        pieces = len(p.get("subset_S", []))
        tikz = ascii_to_tikz(p["ascii"])
        entry = (
            "\\begin{center}\n"
            f"Pieces: {pieces} -- Hash: {h}\\\\[0.5ex]\n"
            f"{tikz}\n"
            "\\end{center}"
            "\\vspace{0.5cm}"
        )
        entries.append(entry)

    doc: List[str] = [
        "\\documentclass[a4paper]{article}",
        "\\usepackage{tikz}",
        "\\usepackage{xcolor}",
        "\\usepackage[margin=1.5cm]{geometry}",
        "\\begin{document}",
    ]

    doc.extend(entries)
    doc.append("\\end{document}")
    return "\n".join(doc)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create printable puzzle sheets")
    parser.add_argument("input", type=Path, help="JSON file with puzzle list")
    parser.add_argument("output", type=Path, help="Output .tex file path")
    args = parser.parse_args()

    puzzles = json.loads(args.input.read_text())
    tex = puzzles_to_latex(puzzles)
    args.output.write_text(tex)


if __name__ == "__main__":
    main()
