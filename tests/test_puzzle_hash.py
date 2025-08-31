import sys
from pathlib import Path

# Ensure the project root is on the path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ubongo import puzzle_hash, parse_polyomino


def test_ascii_and_cells_inputs_same_hash():
    cells = parse_polyomino("square", "##\n##").base
    ascii_art = "##\n##"
    assert puzzle_hash(cells=cells) == puzzle_hash(ascii_art=ascii_art)


def test_different_puzzles_different_hashes():
    h_square = puzzle_hash(ascii_art="##\n##")
    h_lshape = puzzle_hash(ascii_art="#.\n##")
    assert h_square != h_lshape
