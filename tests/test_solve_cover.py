import random
import sys
from pathlib import Path

# Ensure the project root is on the path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ubongo import parse_polyomino, solve_cover


def test_solutions_detail_tracks_piece_placements():
    shape = frozenset({(0, 0), (1, 0), (0, 1), (1, 1)})
    d1 = parse_polyomino("D1", "##")
    d2 = parse_polyomino("D2", "##")

    random.seed(0)
    stats = solve_cover(shape, [d1, d2], required=set(), limit=10)

    assert stats.solutions == 4
    assert stats.nodes == 8
    assert len(stats.solutions_detail) == stats.solutions

    horizontal = frozenset({(0, 0), (1, 0)})
    vertical = frozenset({(0, 0), (0, 1)})
    expected_names = {"D1", "D2"}

    for sol in stats.solutions_detail:
        assert len(sol) == 2
        names = {p[0] for p in sol}
        assert names == expected_names
        covered = set()
        for name, orient, offset in sol:
            assert orient in (horizontal, vertical)
            ox, oy = offset
            for x, y in orient:
                covered.add((x + ox, y + oy))
        assert covered == shape
