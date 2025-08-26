import sys
from pathlib import Path

# Ensure the project root is on the path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ubongo import generate_puzzle_with_mandatory_alt, PIECES


def test_puzzle_generation_is_seed_deterministic():
    kwargs = dict(
        library=PIECES,
        k=3,
        max_w=8,
        max_h=6,
        mandatory_piece="I3",
        seed=123,
        max_attempts=100,
    )
    p1 = generate_puzzle_with_mandatory_alt(**kwargs)
    p2 = generate_puzzle_with_mandatory_alt(**kwargs)
    assert p1 == p2
