import matplotlib.pyplot as plt
import numpy as np
from ubongo import PIECES, generate_puzzle_with_mandatory_alt


def main():
    ratios = []
    puzzles = []
    for seed in range(100):
        print(f"Probing puzzle generation for seed {seed}")
        puzzle = generate_puzzle_with_mandatory_alt(
            library=PIECES,
            k=5,
            max_w=8,
            max_h=6,
            mandatory_piece="I3",
            seed=seed,
            max_attempts=600,
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

    # Show puzzles and their seed if they are in top 10% of circumference ratios
    top_ratios = np.percentile(ratios, 10)
    for i in range(len(puzzles)):
        seed = puzzles[i][0]
        puzzle = puzzles[i][1]
        if puzzle["circumference_ratio"]<=top_ratios:
            print(f"Puzzle found for seed {seed} with ratio {puzzle['circumference_ratio']}")
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

if __name__ == "__main__":
    main()
