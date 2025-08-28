import sys
from pathlib import Path

# Ensure the project root is on the path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ubongo import parse_polyomino, shape_circumference, circumference_ratio


def test_circumference_and_ratio_square():
    shape = parse_polyomino("square", "##\n##").base
    assert shape_circumference(shape) == 8
    assert circumference_ratio(shape) == 2.0
