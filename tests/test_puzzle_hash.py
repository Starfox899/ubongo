import sys
from pathlib import Path

# Ensure the project root is on the path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ubongo import puzzle_hash, parse_polyomino, cells_to_ascii


def test_cells_to_ascii_and_hash():
    cells = parse_polyomino("square", "##\n##").base
    assert cells_to_ascii(cells) == "##\n##"
    assert puzzle_hash(cells_to_ascii(cells)) == puzzle_hash("##\n##")


def test_different_puzzles_different_hashes():
    h_square = puzzle_hash("##\n##")
    h_lshape = puzzle_hash("#.\n##")
    assert h_square != h_lshape
