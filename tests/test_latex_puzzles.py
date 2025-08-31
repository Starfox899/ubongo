from print_puzzles import puzzles_to_latex
from ubongo import puzzle_hash


def test_dedup_and_annotations():
    puzzles = [
        {"ascii": "##\n##", "subset_S": ["A", "B", "C"]},
        {"ascii": "##\n##", "subset_S": ["X"]},  # duplicate shape
        {"ascii": "#.\n##", "subset_S": ["Y", "Z"]},
    ]
    tex = puzzles_to_latex(puzzles)

    # Two unique puzzles -> two hashes
    assert tex.count("Hash:") == 2

    # Hash of square appears and pieces count taken from first occurrence
    h_square = puzzle_hash("##\n##")
    assert f"Hash: {h_square}" in tex
    assert "Pieces: 3" in tex  # from first entry

    # Ensure scaling is correct
    assert "x=12.5mm" in tex
