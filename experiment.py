import json
import matplotlib.pyplot as plt
import numpy as np
from ubongo import PIECES, generate_puzzle_with_mandatory_alt


def main():
    ratios = []
    puzzles = []
    for seed in range(500):
        print(f"Probing puzzle generation for seed {seed}")
        puzzle = generate_puzzle_with_mandatory_alt(
            library=PIECES,
            k=5,
            max_w=6,
            max_h=4,
            mandatory_piece="I3",
            seed=seed,
            max_attempts=1000,
        )
        if puzzle is not None:
            ratios.append(puzzle["circumference_ratio"])
            puzzles.append((seed,puzzle))
        else:
            print(f"No puzzle found for seed {seed}; skipping.")

    plt.hist(ratios, bins="auto", edgecolor="black")
    plt.title("Circumference Ratio Distribution for 100 Ubongo Puzzles")
    plt.xlabel("Circumference / Area ratio")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Show puzzles and collect those in the top X% of circumference ratios
    top_ratios = np.percentile(ratios, 100)
    best: list[dict] = []
    for seed, puzzle in puzzles:
        if puzzle["circumference_ratio"] <= top_ratios:
            print(
                f"Puzzle found for seed {seed} with ratio {puzzle['circumference_ratio']}"
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
            best.append({"ascii": puzzle["ascii"], "subset_S": puzzle["subset_S"]})

    # Persist selected puzzles for later printing
    with open("puzzles.json", "w") as fh:
        json.dump(best, fh, indent=2)

if __name__ == "__main__":
    main()
